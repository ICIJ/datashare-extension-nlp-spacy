import multiprocessing
from typing import Dict, List, Optional

from icij_worker.typing_ import RateProgress
from spacy import Language

from spacy_worker.core import spacy_ner as spacy_ner_
from spacy_worker.objects import Category, SpacySize
from spacy_worker.tasks.dependencies import lifespan_config, lifespan_spacy_provider

_DEFAULT_CATEGORIES = [Category.LOC, Category.PER, Category.ORG]


async def spacy_ner(
    texts: List[str],
    language: str,
    *,
    categories: List[str] = None,
    size: str = SpacySize.SMALL.value,
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


def get_n_process(ner: Language, max_processes: int) -> int:
    if "transformer" in ner.pipe_names:
        return 1
    if max_processes == -1:
        max_processes = multiprocessing.cpu_count()
    return max_processes - 1  # For the sentence splitter
