from typing import Dict, List, Optional

from icij_worker import AsyncApp
from icij_worker.typing_ import RateProgress

from spacy_worker.tasks import spacy_ner as spacy_ner_
from spacy_worker.tasks.dependencies import APP_LIFESPAN_DEPS

app = AsyncApp("spacy", dependencies=APP_LIFESPAN_DEPS)


@app.task(name="spacy-ner")
async def spacy_ner(
    texts: List[str],
    language: str,
    *,
    categories: List[str] = None,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    return await spacy_ner_(
        texts, language=language, categories=categories, progress=progress
    )
