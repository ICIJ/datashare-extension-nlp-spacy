from typing import ClassVar, List

from icij_common.pydantic_utils import ICIJSettings, NoEnumModel
from icij_worker.utils.logging_ import LogWithWorkerIDMixin
from pydantic import Field

import spacy_worker
from spacy_worker.core import SpacyProvider

_ALL_LOGGERS = [spacy_worker.__name__]


class AppConfig(ICIJSettings, LogWithWorkerIDMixin, NoEnumModel):
    class Config:
        env_prefix = "DS_DOCKER_SPACY_"

    loggers: ClassVar[List[str]] = Field(_ALL_LOGGERS, const=True)

    log_level: str = Field(default="INFO")

    batch_size: int = 1024
    max_processes: int = -1
    max_languages_in_memory: int = 2

    def to_provider(self) -> SpacyProvider:
        return SpacyProvider(self.max_languages_in_memory)
