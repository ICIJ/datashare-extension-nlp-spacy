import logging
from typing import Dict, List, Optional

from icij_common.logging_utils import log_elapsed_time_cm
from icij_worker import AsyncApp
from icij_worker.typing_ import RateProgress

from spacy_worker.constants import DS_SPACY_NER_TASK, SPACY_NER_TASK
from spacy_worker.objects import SpacySize
from spacy_worker.tasks import ds_spacy_ner as ds_spacy_ner_, spacy_ner as spacy_ner_
from spacy_worker.tasks.dependencies import APP_LIFESPAN_DEPS

app = AsyncApp("spacy", dependencies=APP_LIFESPAN_DEPS)

logger = logging.getLogger(__name__)


@app.task(name=SPACY_NER_TASK)
async def spacy_ner(
    docs: List[Dict],
    language: str,
    *,
    size: str = SpacySize.SMALL.value,
    categories: List[str] = None,
    progress: Optional[RateProgress] = None,
) -> List[Dict]:
    msg = f"worker processed {len(docs)} docs in {{elapsed_time}} !"
    with log_elapsed_time_cm(logger, level=logging.INFO, output_msg=msg):
        return await spacy_ner_(
            docs, language=language, categories=categories, progress=progress, size=size
        )


@app.task(name=DS_SPACY_NER_TASK)
async def ds_spacy_ner(
    docs: List[Dict],
    *,
    size: str = SpacySize.SMALL.value,
    categories: List[str] = None,
    progress: Optional[RateProgress] = None,
) -> List[List[Dict]]:
    msg = f"worker processed {len(docs)} docs in {{elapsed}} !"
    with log_elapsed_time_cm(logger, level=logging.INFO, output_msg=msg):
        return await ds_spacy_ner_(
            docs, size=size, categories=categories, progress=progress
        )
