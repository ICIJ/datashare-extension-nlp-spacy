import logging
from multiprocessing import Pool
from typing import Optional

from icij_worker import WorkerConfig
from icij_worker.utils.dependencies import DependencyInjectionError

from spacy_worker.config import AppConfig
from spacy_worker.core import SpacyProvider

logger = logging.getLogger(__name__)

_ASYNC_APP_CONFIG: Optional[AppConfig] = None
_SPACY_PROVIDER: Optional[Pool] = None


def load_app_config(worker_config: WorkerConfig, **_):
    global _ASYNC_APP_CONFIG
    if worker_config.app_bootstrap_config_path is not None:
        _ASYNC_APP_CONFIG = AppConfig.parse_file(
            worker_config.app_bootstrap_config_path
        )
    else:
        _ASYNC_APP_CONFIG = AppConfig()


def setup_loggers(worker_id: str, **_):
    config = lifespan_config()
    config.setup_loggers(worker_id=worker_id)
    logger.info("worker loggers ready to log 💬")
    logger.info("app config: %s", config.json(indent=2))


def lifespan_config() -> AppConfig:
    if _ASYNC_APP_CONFIG is None:
        raise DependencyInjectionError("config")
    return _ASYNC_APP_CONFIG


def spacy_provider_enter(**_):
    config = lifespan_config()
    global _SPACY_PROVIDER
    _SPACY_PROVIDER = config.to_provider()
    _SPACY_PROVIDER.__enter__()


def spacy_provider_exit(exc_type, exc_val, exc_tb):
    config = lifespan_config()
    global _SPACY_PROVIDER
    _SPACY_PROVIDER = config.to_provider()
    _SPACY_PROVIDER.__exit__(exc_type, exc_val, exc_tb)


def lifespan_spacy_provider() -> SpacyProvider:
    if _SPACY_PROVIDER is None:
        raise DependencyInjectionError("spacy provider")
    return _SPACY_PROVIDER


APP_LIFESPAN_DEPS = [
    ("loading async app configuration", load_app_config, None),
    ("loggers", setup_loggers, None),
    ("spacy provider", spacy_provider_enter, spacy_provider_exit),
]
