from typing import Literal

from icij_worker import AsyncApp
from icij_worker.app import TaskGroup
from icij_worker.typing_ import RateProgress

from datashare_spacy_worker.tasks import spacy_ner_task as spacy_ner_
from datashare_spacy_worker.tasks.dependencies import APP_LIFESPAN_DEPS

app = AsyncApp("spacy", dependencies=APP_LIFESPAN_DEPS)
PYTHON_TASK_GROUP = "PYTHON"
_SPACY_PIPELINE = "SPACY"

@app.task(name="BatchNlp", group=TaskGroup(name=PYTHON_TASK_GROUP))
async def spacy_ner(
    docs: list[dict],
    categories: list[str] = None,
    *,
    model_size: str | None = None,
    max_length: int,
    progress: RateProgress | None = None,
    pipeline: Literal["SPACY"],
) -> int:
    if pipeline != _SPACY_PIPELINE:
        raise ValueError(f"invalid pipeline: {pipeline} expected {_SPACY_PIPELINE}")
    return await spacy_ner_(
        docs,
        categories=categories,
        model_size=model_size,
        progress=progress,
        max_length=max_length,
    )
