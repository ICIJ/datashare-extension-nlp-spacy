from typing import Dict, List, Optional

from icij_worker import AsyncApp
from icij_worker.typing_ import RateProgress

from spacy_worker.objects import SpacySize
from spacy_worker.tasks import spacy_ner as spacy_ner_, ds_spacy_ner as ds_spacy_ner_
from spacy_worker.tasks.dependencies import APP_LIFESPAN_DEPS

app = AsyncApp("spacy", dependencies=APP_LIFESPAN_DEPS)


@app.task(name="spacy-ner")
async def spacy_ner(
    texts: List[str],
    language: str,
    *,
    size: str = SpacySize.SMALL.value,
    categories: List[str] = None,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    return await spacy_ner_(
        texts, language=language, categories=categories, progress=progress, size=size
    )


@app.task(name="ds-spacy-ner")
async def ds_spacy_ner(
    docs: List[Dict],
    *,
    size: str = SpacySize.SMALL.value,
    categories: List[str] = None,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    return await ds_spacy_ner_(
        docs, size=size, categories=categories, progress=progress
    )
