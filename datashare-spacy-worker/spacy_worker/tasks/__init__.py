import functools
import multiprocessing
from typing import Callable, Dict, List, Optional, Set

from icij_common.pydantic_utils import LowerCamelCaseModel, jsonable_encoder
from icij_worker.typing_ import RateProgress
from icij_worker.utils.progress import to_raw_progress
from pydantic.tools import parse_obj_as
from spacy import Language

from spacy_worker.core import SpacyProvider, spacy_ner as spacy_ner_
from spacy_worker.objects import Category, NamedEntity, SpacySize
from spacy_worker.tasks.dependencies import lifespan_config, lifespan_spacy_provider

_DEFAULT_CATEGORIES = [Category.LOC, Category.PER, Category.ORG]


async def spacy_ner(
    texts: List[str],
    language: str,
    *,
    categories: List[str] = None,
    size: str,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    if categories is None:
        categories = set(c.value for c in _DEFAULT_CATEGORIES)
    else:
        categories = set(Category(c) for c in categories)
    config = lifespan_config()
    spacy_provider = lifespan_spacy_provider()
    size = SpacySize(size)
    ner = spacy_provider.get_ner(language, size=size)
    sent_split = spacy_provider.get_sent_split(language)
    n_process = get_n_process(ner, max_processes=config.max_processes)
    entities_gen = spacy_ner_(
        texts,
        ner,
        categories=categories,
        sent_split=sent_split,
        n_process=n_process,
        progress=progress,
        batch_size=config.batch_size,
    )
    entities = [[e.dict() for e in ents] async for ents in entities_gen]
    return entities


class DSDoc(LowerCamelCaseModel):
    id: str
    project: str
    root_id: str


async def ds_spacy_ner(
    docs: List[Dict],
    categories: List[str] = None,
    *,
    size: str,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    if categories is None:
        categories = set(c.value for c in _DEFAULT_CATEGORIES)
    else:
        categories = set(Category(c) for c in categories)
    # Note: we don't save NE directly in ES as this would imply a very tight coupling
    # between the Python worker which doesn't depend on the Java API. If the Named
    # Entity format change, the Python worker can still NE in the old format without
    # any error. Error will occur only late in the process when they are loaded on the
    # Java side...
    docs = parse_obj_as(List[Dict], docs)
    n_docs = len(docs)
    if progress is not None:
        progress = to_raw_progress(progress, max_progress=n_docs)
    config = lifespan_config()
    es_client = lifespan_es_client()
    spacy_provider = lifespan_spacy_provider()
    size = SpacySize(size)
    max_content_length = config.max_content_length
    batch_size = config.batch_size
    # TODO: like in Java we retrieve docs 1 by 1 to avoid loading several huge docs
    #  content in memory
    batch = []
    # Careful, this could get real large...
    tags = [[] for _ in range(n_docs)]
    current_language = None
    batch_ix_to_doc_ix = dict()
    process_fn = None
    for doc_i, doc in enumerate(docs):
        es_doc = await es_client.get_source(
            index=doc.project,
            id=doc.id,
            routing=doc.root_id,
            _source_includes=["language", "content"],
        )
        language = es_doc["language"]
        if language != current_language:
            if batch:
                _consume_batch(process_fn, batch, batch_ix_to_doc_ix, tags)
            process_fn = _get_process_fn(
                spacy_provider,
                size,
                categories,
                language,
                max_process=config.max_processes,
                batch_size=batch_size,
            )
        doc_content = es_doc["content"]
        doc_length = len(doc_content)
        for text_i in range(0, doc_length, max_content_length):
            if len(batch) >= batch_size:
                _consume_batch(process_fn, batch, batch_ix_to_doc_ix, tags)
                if progress is not None:
                    await progress(doc_i)
            chunk = doc_content[text_i : text_i + max_content_length]
            batch.append(chunk)
            batch_ix_to_doc_ix[len(batch) - 1] = doc_i
    if len(batch) >= batch_size:
        _consume_batch(process_fn, batch, batch_ix_to_doc_ix, tags)
        if progress is not None:
            await progress(n_docs)
    tags = jsonable_encoder(tags)
    return tags


def _get_process_fn(
    spacy_provider: SpacyProvider,
    size: SpacySize,
    categories: Set[Category],
    language: str,
    *,
    max_process: int,
    batch_size: int,
):
    ner = spacy_provider.get_ner(language, size=size)
    sent_split = spacy_provider.get_sent_split(language)
    n_process = get_n_process(ner, max_processes=max_process)
    process_fn = functools.partial(
        spacy_ner_,
        ner=ner,
        categories=categories,
        sent_split=sent_split,
        n_process=n_process,
        progress=None,
        batch_size=batch_size,
    )
    return process_fn


def _consume_batch(
    _process_batch: Callable,
    batch: List[str],
    batch_ix_to_doc_ix: Dict[int, int],
    tags: List[List[NamedEntity]],
):
    batch_tags = _process_batch(batch)
    for c_i, chunk_tags in enumerate(batch_tags):
        tags[batch_ix_to_doc_ix[c_i]].extend(chunk_tags)
    batch_tags.clear()
    batch_ix_to_doc_ix.clear()


def get_n_process(ner: Language, max_processes: int) -> int:
    if "transformer" in ner.pipe_names:
        return 1
    if max_processes == -1:
        max_processes = multiprocessing.cpu_count()
    return max_processes - 1  # For the sentence splitter
