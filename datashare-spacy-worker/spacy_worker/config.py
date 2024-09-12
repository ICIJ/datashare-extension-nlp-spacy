from typing import ClassVar, List, Optional, Union

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

    # DS
    ds_api_key: Optional[str] = None
    # ES
    es_address: str
    es_default_page_size: int = 1000
    es_keep_alive: str = "1m"
    es_max_concurrency: int = 5
    es_max_retries: int = 0
    es_max_retry_wait_s: Union[int, float] = 60
    es_timeout_s: Union[int, float] = 60 * 5

    max_content_length: int = 1024
    max_processes: int = -1
    max_languages_in_memory: int = 2

    def to_provider(self) -> SpacyProvider:
        return SpacyProvider(self.max_languages_in_memory)

    def to_es_client(self, address: Optional[str] = None) -> "ESClient":
        from spacy_worker.es import ESClient

        if address is None:
            address = self.es_address

        client = ESClient(
            hosts=[address],
            pagination=self.es_default_page_size,
            max_concurrency=self.es_max_concurrency,
            keep_alive=self.es_keep_alive,
            timeout=self.es_timeout_s,
            max_retries=self.es_max_retries,
            max_retry_wait_s=self.es_max_retry_wait_s,
            api_key=self.ds_api_key,
        )
        client.transport._verified_elasticsearch = True
        return client
