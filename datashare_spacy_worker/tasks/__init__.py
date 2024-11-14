import functools
import multiprocessing

import math
import pycountry
from elasticsearch._async.helpers import async_bulk
from icij_common.es import DOC_CONTENT, DOC_LANGUAGE, ESClient, ID_
from icij_common.pydantic_utils import jsonable_encoder
from icij_worker.typing_ import RateProgress
from icij_worker.utils.progress import to_raw_progress
from pydantic import parse_obj_as
from spacy import Language
from spacy.tokens.doc import defaultdict

from datashare_spacy_worker.core import spacy_ner as spacy_ner_
from datashare_spacy_worker.objects import (
    BatchDocument,
    Category,
    NamedEntity,
    NlpTag,
    SpacySize,
)
from datashare_spacy_worker.tasks.dependencies import (
    lifespan_config,
    lifespan_es_client,
    lifespan_spacy_provider,
)

_DEFAULT_CATEGORIES = [Category.LOC, Category.PER, Category.ORG]
_NE_SCHEMA = NamedEntity.schema(by_alias=True)


async def spacy_ner_task(
    docs: list[dict],
    categories: list[str] = None,
    *,
    model_size: str,
    max_length: int,
    progress: RateProgress | None = None,
) -> int:
    if not docs:
        return len(docs)
    if categories is None:
        categories = set(c.value for c in _DEFAULT_CATEGORIES)
    else:
        categories = set(Category(c) for c in categories)
    docs = parse_obj_as(list[BatchDocument], docs)
    if progress is not None:
        progress = to_raw_progress(progress, max_progress=len(docs))
    config = lifespan_config()
    es_client = lifespan_es_client()
    # TODO: perform a NamedEntityModel check
    spacy_provider = lifespan_spacy_provider()
    model_size = SpacySize(model_size)
    docs_language = docs[0].language
    language = pycountry.languages.get(name=docs_language)
    if language is None or not hasattr(language, "alpha_2"):
        raise ValueError(f'Unknown language "{docs_language}"')
    language = language.alpha_2
    ner = spacy_provider.get_ner(language, model_size=model_size)
    sent_split = spacy_provider.get_sent_split(language, model_size=model_size)
    n_process = get_n_process(ner, max_processes=config.max_processes)
    return await _process_docs(
        docs,
        ner,
        sent_split,
        categories,
        es_client,
        pipeline_batch_size=config.pipeline_batch_size,
        batch_size=config.batch_size,
        ne_buffer_size=config.ne_buffer_size,
        max_length=max_length,
        n_process=n_process,
        progress=progress,
    )


class _TagsBuffer:
    def __init__(self, max_length: int):
        self._max_length = max_length
        self._batch_ix_to_doc_ix: dict[int, int] = dict()
        self._n_doc_chunks: dict[int, int] = dict()
        self._doc_tags: dict[int, list[list[NlpTag]]] = defaultdict(list)

    def add_doc(self, i: int, doc: dict):
        self._n_doc_chunks[i] = math.ceil(len(doc[DOC_CONTENT]) / self._max_length)

    def add_batch_tags(self, batch_tags: list[list[NlpTag]]):
        for tag_i, tags in enumerate(batch_tags):
            chunks_tags = self._doc_tags[self._batch_ix_to_doc_ix[tag_i]]
            offset = self._max_length * len(chunks_tags)
            tags = [t.with_offset(offset) for t in tags]
            chunks_tags.append(tags)
        for doc_i, tags in self._doc_tags.items():
            if len(tags) != self._n_doc_chunks[doc_i]:
                continue
        self._batch_ix_to_doc_ix.clear()

    def set_batch_doc_ix(self, doc_ix: int):
        new_ix = len(self._batch_ix_to_doc_ix)
        self._batch_ix_to_doc_ix[new_ix] = doc_ix

    def get_ready(self) -> dict[int, list[NlpTag]]:
        ready = {
            doc_i: sum(tags, start=[])
            for doc_i, tags in self._doc_tags.items()
            if len(tags) == self._n_doc_chunks[doc_i]
        }
        for doc_i in ready:
            self._n_doc_chunks.pop(doc_i)
            self._doc_tags.pop(doc_i)
        return ready

    def __len__(self):
        return len(self._n_doc_chunks)


async def _process_docs(
    docs: list[BatchDocument],
    ner,
    sent_split,
    categories: set[Category],
    es_client: ESClient,
    *,
    batch_size: int,
    pipeline_batch_size: int,
    max_length: int,
    ne_buffer_size: int,
    n_process: int,
    progress: RateProgress | None,
):
    # pylint: disable=not-an-iterable
    batch = []
    n_docs = len(docs)
    if not n_docs:
        return
    project = docs[0].project
    tags_buffer = _TagsBuffer(max_length)
    ne_buffer = []
    tag_fn = functools.partial(
        spacy_ner_,
        ner=ner,
        categories=categories,
        sent_split=sent_split,
        n_process=n_process,
        progress=None,
        batch_size=pipeline_batch_size,
    )
    # Let's not overload the broker with events and only log progress halfway
    progress_rate = len(docs) / 2
    for doc_i, doc in enumerate(docs):
        es_doc = await es_client.get_source(
            index=doc.project,
            id=doc.id,
            routing=doc.root_document,
            _source_includes=[DOC_LANGUAGE, DOC_CONTENT],
        )
        tags_buffer.add_doc(doc_i, es_doc)
        doc_content = es_doc[DOC_CONTENT]
        doc_length = len(doc_content)
        for text_i in range(0, doc_length, max_length):
            chunk = doc_content[text_i : text_i + max_length]
            batch.append(chunk)
            tags_buffer.set_batch_doc_ix(doc_i)
            if len(batch) >= batch_size:
                batch_tags = [text_tags async for text_tags in tag_fn(batch)]
                ready_ents = await _update_and_consume_buffer(
                    tags_buffer, batch_tags, docs
                )
                ne_buffer += ready_ents
                if len(ne_buffer) >= ne_buffer_size:
                    await _bulk_add_ne(es_client, ne_buffer, project=project)
                    ne_buffer.clear()
                batch.clear()
        if progress is not None and not doc_i % progress_rate:
            await progress(doc_i / n_docs)
    if batch:
        batch_tags = [text_tags async for text_tags in tag_fn(batch)]
        ready_ents = await _update_and_consume_buffer(tags_buffer, batch_tags, docs)
        if len(tags_buffer):
            raise ValueError(
                "inconsistent state: tags buffer was not empty processing all docs"
            )
        ne_buffer += ready_ents
    if ne_buffer:
        await _bulk_add_ne(es_client, ne_buffer, project=project)
    return n_docs


async def _update_and_consume_buffer(
    tags_buffer: _TagsBuffer,
    batch_tags: list[list[NlpTag]],
    docs: list[BatchDocument],
) -> list[list[NamedEntity]]:
    tags_buffer.add_batch_tags(batch_tags)
    ready_ents = sum(
        (
            NamedEntity.from_tags(tags, docs[ready_i])
            for ready_i, tags in tags_buffer.get_ready().items()
        ),
        [],
    )
    return ready_ents


async def _bulk_add_ne(
    es_client: ESClient, named_entities: list[NamedEntity], project: str
):
    named_entities = jsonable_encoder(named_entities)
    actions = (
        {
            "_op_type": "update",
            "_index": project,
            ID_: ne["id"],
            "doc": ne,
            "doc_as_upsert": True,
        }
        for ne in named_entities
    )
    await async_bulk(es_client, actions, raise_on_error=True, refresh="wait_for")


def get_n_process(ner: Language, max_processes: int) -> int:
    if "transformer" in ner.pipe_names:
        return 1
    if max_processes == -1:
        max_processes = multiprocessing.cpu_count()
    return max_processes
