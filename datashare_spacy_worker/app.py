from icij_worker import AsyncApp
from icij_worker.typing_ import RateProgress

from datashare_spacy_worker.tasks import spacy_ner_task as spacy_ner_
from datashare_spacy_worker.tasks.dependencies import APP_LIFESPAN_DEPS

app = AsyncApp("spacy", dependencies=APP_LIFESPAN_DEPS)


@app.task(name="spacy-ner")
async def spacy_ner(
    docs: list[dict],
    categories: list[str] = None,
    *,
    model_size: str | None = None,
    max_length: int,
    progress: RateProgress | None = None,
) -> int:
    return await spacy_ner_(
        docs,
        categories=categories,
        model_size=model_size,
        progress=progress,
        max_length=max_length,
    )
