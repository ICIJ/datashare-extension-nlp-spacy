import logging
from multiprocessing import Pool
from types import TracebackType
from typing import Optional

from icij_common.es import ESClient
from icij_worker import WorkerConfig
from icij_worker.utils.dependencies import DependencyInjectionError

from datashare_spacy_worker.config import AppConfig
from datashare_spacy_worker.core import SpacyProvider

logger = logging.getLogger(__name__)

_ASYNC_APP_CONFIG: AppConfig | None = None
_SPACY_PROVIDER: Optional[Pool] = None
_ES_CLIENT: ESClient | None = None


def load_app_config(worker_config: WorkerConfig, **_) -> None:
    global _ASYNC_APP_CONFIG
    if worker_config.app_bootstrap_config_path is not None:
        _ASYNC_APP_CONFIG = AppConfig.model_validate_json(
            worker_config.app_bootstrap_config_path.read_text()
        )
    else:
        _ASYNC_APP_CONFIG = AppConfig()


def setup_loggers(worker_id: str, **_) -> None:
    config = lifespan_config()
    config.setup_loggers(worker_id=worker_id)
    logger.info("worker loggers ready to log 💬")
    logger.info("app config: %s", config.model_dump_json(indent=2))


def lifespan_config() -> AppConfig:
    if _ASYNC_APP_CONFIG is None:
        raise DependencyInjectionError("config")
    return _ASYNC_APP_CONFIG


def spacy_provider_enter(**_) -> None:
    config = lifespan_config()
    global _SPACY_PROVIDER
    _SPACY_PROVIDER = config.to_provider()
    _SPACY_PROVIDER.__enter__()


def spacy_provider_exit(
    exc_type: type[Exception], exc_val: Exception, exc_tb: TracebackType
) -> None:
    lifespan_spacy_provider().__exit__(exc_type, exc_val, exc_tb)
    global _SPACY_PROVIDER
    _SPACY_PROVIDER = None


def lifespan_spacy_provider() -> SpacyProvider:
    if _SPACY_PROVIDER is None:
        raise DependencyInjectionError("spacy provider")
    return _SPACY_PROVIDER


async def es_client_enter(**_) -> None:
    config = lifespan_config()
    global _ES_CLIENT
    _ES_CLIENT = config.to_es_client()
    await _ES_CLIENT.__aenter__()


async def es_client_exit(
    exc_type: type[Exception], exc_val: Exception, exc_tb: TracebackType
) -> None:
    await lifespan_es_client().__aexit__(exc_type, exc_val, exc_tb)
    global _ES_CLIENT
    _ES_CLIENT = None


def lifespan_es_client() -> ESClient:
    if _ES_CLIENT is None:
        raise DependencyInjectionError("es client")
    return _ES_CLIENT


APP_LIFESPAN_DEPS = [
    ("loading async app configuration", load_app_config, None),
    ("loggers", setup_loggers, None),
    ("spacy provider", spacy_provider_enter, spacy_provider_exit),
    ("elasticsearch client", es_client_enter, es_client_exit),
]
