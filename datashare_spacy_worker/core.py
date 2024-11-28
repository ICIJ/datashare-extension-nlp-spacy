from __future__ import annotations

import json
import logging
import shutil
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator
from urllib.parse import urljoin

import spacy
from icij_worker.typing_ import RateProgress
from icij_worker.utils.progress import to_raw_progress
from spacy import Language as SpacyLanguage, about as spacy_about
from spacy.cli.download import (
    get_compatibility,
    get_model_filename,
    get_version,
)
from spacy.tokens import Doc as SpacyDoc, Span
from spacy.util import load_model_from_init_py

from datashare_spacy_worker.constants import DATA_DIR
from datashare_spacy_worker.ner_label_scheme import NERLabelScheme
from datashare_spacy_worker.objects import Category, NlpTag, SpacySize

logger = logging.getLogger(__name__)

DocCtx = dict[str, Any]
_DEFAULT_EXCLUDE = [
    "tagger",
    "morphologizer",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "senter",
]


async def spacy_ner(
    text: list[str],
    ner: SpacyLanguage,
    *,
    categories: list[Category],
    sent_split: SpacyLanguage,
    n_process: int = -1,
    batch_size: int | None,
    progress: RateProgress | None = None,
) -> Generator[list[NlpTag], None, None]:
    categories = set(categories)
    if progress is not None:
        progress = to_raw_progress(progress, max_progress=len(text))
    label_scheme = NERLabelScheme(tuple(ner.pipe_labels.get("ner")))
    # TODO: for better NER performance, it could be nicer have chunks for several
    #  sentence rather than just sentence by sentence
    split_texts = _split_docs_spacy(text, sent_split)
    previous_ctx = None
    sub_docs = []
    logger.debug("creating spacy pipe with %s process(es)", n_process)
    pipe = ner.pipe(
        split_texts, n_process=n_process, as_tuples=True, batch_size=batch_size
    )
    n_processed_texts = 0
    if progress is not None:
        await progress(n_processed_texts)
    for doc, ctx in pipe:
        if previous_ctx is not None and ctx["doc_ix"] != previous_ctx["doc_ix"]:
            merged = _merge_subdocs(sub_docs)
            predicted = _spacy_doc_to_ds_tags(merged, categories, label_scheme)
            yield predicted
            n_processed_texts += 1
            update_progress = not ctx["doc_ix"] % 50
            if progress is not None and update_progress:
                await progress(n_processed_texts)
            sub_docs = [doc]
        else:
            sub_docs.append(doc)
        previous_ctx = ctx
    if sub_docs:
        merged = _merge_subdocs(sub_docs)
        predicted = _spacy_doc_to_ds_tags(merged, categories, label_scheme)
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

    def get_ner(self, language: str, *, model_size: SpacySize) -> SpacyLanguage:
        return self._load_nlp(language, model_size=model_size)

    def get_sent_split(self, language: str, *, model_size: SpacySize) -> SpacyLanguage:
        vocab = self._load_nlp(language, model_size=model_size).vocab
        sent_split = spacy.blank(language, vocab=vocab)
        sent_split.add_pipe("sentencizer")
        return sent_split

    def _load_nlp(self, language: str, *, model_size: SpacySize) -> SpacyLanguage:
        # pylint: disable=method-hidden
        logger.debug("loading spacy for %s...", language)
        model_size = model_size.value
        model = self._models[language]["sizes"][model_size]
        model_name = f"{language}_{model['model']}_{model_size}"
        # TODO: use GPU acceleration using spacy.prefer_gpu() + spacy[cuda]
        # TODO: check if we can exclude globally or we must do it language per langauge
        exclude = model.get("exclude", _DEFAULT_EXCLUDE)
        # We have to do some dark magic because of pyinstaller...
        model_path = DATA_DIR / model_name
        if not model_path.exists():
            logger.info("downloading spacy model  %s...", model_name)
            # Hack due to the fact pyinstaller doesn't support a python script from
            # the pyinstaller binary. Sadly this is the case when installing a spacy
            # model...
            _fixed_pysinstaller_download(model_name, model_path)
        return load_model_from_init_py(model_path / "__init__.py", exclude=exclude)


def _merge_subdocs(sub_docs: list[SpacyDoc]) -> SpacyDoc:
    return SpacyDoc.from_docs(sub_docs, ensure_whitespace=False)


def _split_docs_spacy(
    docs: Iterable[str], sent_split: SpacyLanguage
) -> Iterator[tuple[SpacyDoc, DocCtx]]:
    inputs = ((doc, {"doc_ix": i}) for i, doc in enumerate(docs))
    return sent_split.pipe(inputs, as_tuples=True)


def _spacy_doc_to_ds_tags(
    doc: SpacyDoc, supported_categories: set[Category], label_scheme: NERLabelScheme
) -> list[NlpTag]:
    ents = (
        _spacy_doc_to_ds_tag(ent, supported_categories, label_scheme)
        for ent in doc.ents
    )
    return [ent for ent in ents if ent is not None]


def _spacy_doc_to_ds_tag(
    ent: Span,
    supported_categories: set[Category],
    label_scheme: NERLabelScheme,
) -> NlpTag | None:
    category = label_scheme.to_spacy(ent.label_)
    if category is None or category not in supported_categories:
        return None
    start = ent.start_char
    return NlpTag(start=start, mention=ent.text, category=category)


def _fixed_pysinstaller_download(model_name: str, model_path: Path):
    # Very ugly by pyinstaller forces to do ugly things...
    compatibility = get_compatibility()
    version = get_version(model_name, compatibility)
    filename = get_model_filename(model_name, version)
    base_url = spacy_about.__download_url__
    # urljoin requires that the path ends with /, or the last path part will be
    # dropped
    if not base_url.endswith("/"):
        base_url = spacy_about.__download_url__ + "/"
    download_url = urljoin(base_url, filename)
    wheel_name = Path(filename).name
    wheel_path = model_path.parent / wheel_name
    urllib.request.urlretrieve(download_url, wheel_path)
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(model_path.parent)
    for dir in model_path.parent.iterdir():
        if dir.is_dir() and dir.name.endswith(".dist-info"):
            shutil.rmtree(dir)
