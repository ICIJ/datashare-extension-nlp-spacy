from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import (
    Any,
    Collection,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import spacy
from icij_worker.typing_ import RateProgress
from icij_worker.utils.progress import to_raw_progress
from spacy import Language
from spacy.cli import download
from spacy.tokens import Doc as SpacyDoc, Span
from spacy.util import is_package

from spacy_worker.constants import DATA_DIR
from spacy_worker.ner_label_scheme import NERLabelScheme
from spacy_worker.objects import Category, NamedEntity, SpacySize

logger = logging.getLogger(__name__)

DocCtx = Dict[str, Any]
_DEFAULT_EXCLUDE = [
    "tagger",
    "morphologizer",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "senter",
]


async def spacy_ner(
    texts: Collection[str],
    ner: Language,
    *,
    categories: Set[Category],
    sent_split: Language,
    n_process: int = -1,
    batch_size: Optional[int],
    progress: Optional[RateProgress] = None,
) -> Generator[List[NamedEntity], None, None]:
    if progress is not None:
        progress = to_raw_progress(progress, max_progress=len(texts))
    label_scheme = NERLabelScheme(tuple(ner.pipe_labels.get("ner")))
    # TODO: for better NER performance, it could be nicer have chunks for several
    #  sentence rather than just sentence by sentence
    split_texts = _split_docs_spacy(texts, sent_split)
    previous_ctx = None
    sub_docs = []
    pipe = ner.pipe(
        split_texts, n_process=n_process, as_tuples=True, batch_size=batch_size
    )
    n_processed_texts = 0
    if progress is not None:
        await progress(n_processed_texts)
    for doc, ctx in pipe:
        if previous_ctx is not None and ctx["doc_ix"] != previous_ctx["doc_ix"]:
            merged = _merge_subdocs(sub_docs)
            predicted = _spacy_doc_to_ds_named_entities(
                merged, categories, label_scheme
            )
            yield predicted
            n_processed_texts += 1
            if progress is not None:
                await progress(n_processed_texts)
            sub_docs = [doc]
        else:
            sub_docs.append(doc)
        previous_ctx = ctx
    if sub_docs:
        merged = _merge_subdocs(sub_docs)
        predicted = _spacy_doc_to_ds_named_entities(merged, categories, label_scheme)
        yield predicted
        n_processed_texts += 1
        if progress is not None:
            await progress(n_processed_texts)


class SpacyProvider:
    _model_file_path = DATA_DIR / "models.json"

    def __init__(self, max_languages: int):
        self._max_languages = max_languages
        self._pipelines = dict()
        self._models = json.loads(self._model_file_path.read_text())
        self._load_nlp = lru_cache(maxsize=max_languages)(self._load_nlp)

    def __enter__(self) -> SpacyProvider:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pipelines.clear()

    def get_ner(self, language: str, *, size: SpacySize) -> Language:
        return self._load_nlp(language, size=size)

    def get_sent_split(self, language: str) -> Language:
        vocab = self._load_nlp(language, size=SpacySize.SMALL).vocab
        sent_split = spacy.blank(language, vocab=vocab)
        sent_split.add_pipe("sentencizer")
        return sent_split

    def _load_nlp(self, language: str, *, size: SpacySize) -> Language:
        # pylint: disable=method-hidden
        logger.debug("loading spacy for %s...", language)
        size = size.value
        model = self._models[language]["sizes"][size]
        model_name = f"{language}_{model['model']}_{size}"
        # TODO: use GPU acceleration using spacy.prefer_gpu() + spacy[cuda]
        # TODO: check if we can exclude globally or we must do it language per langauge
        exclude = model.get("exclude", _DEFAULT_EXCLUDE)
        if not is_package(model_name):
            logger.info("downloading spacy model  %s...", model_name)
            download(model_name)
        return spacy.load(model_name, exclude=exclude)


def _merge_subdocs(sub_docs: List[SpacyDoc]) -> SpacyDoc:
    return SpacyDoc.from_docs(sub_docs, ensure_whitespace=False)


def _split_docs_spacy(
    docs: Iterable[str], sent_split: Language
) -> Iterator[Tuple[SpacyDoc, DocCtx]]:
    inputs = ((doc, {"doc_ix": i}) for i, doc in enumerate(docs))
    return sent_split.pipe(inputs, as_tuples=True)


def _spacy_doc_to_ds_named_entities(
    doc: SpacyDoc, supported_categories: Set[Category], label_scheme: NERLabelScheme
) -> List[NamedEntity]:
    ents = (
        _spacy_doc_to_named_entity(ent, supported_categories, label_scheme)
        for ent in doc.ents
    )
    return [ent for ent in ents if ent is not None]


def _spacy_doc_to_named_entity(
    ent: Span,
    supported_categories: Set[Category],
    label_scheme: NERLabelScheme,
) -> Optional[NamedEntity]:
    category = label_scheme.to_spacy(ent.label_)
    if category is None or category not in supported_categories:
        return None
    start = ent.start_char
    end = ent.end_char
    return NamedEntity(start=start, end=end, category=category)
