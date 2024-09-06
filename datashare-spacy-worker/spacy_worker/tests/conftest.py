# pylint: disable=redefined-outer-name
from pathlib import Path

import pytest
from icij_worker import AMQPWorkerConfig

from spacy_worker.app import app
from spacy_worker.config import AppConfig


@pytest.fixture(scope="session")
def test_app_config() -> AppConfig:
    return AppConfig(log_level="DEBUG", max_processes=2)


@pytest.fixture(scope="session")
def test_app_config_path(tmpdir_factory, test_app_config: AppConfig) -> Path:
    config_path = Path(tmpdir_factory.mktemp("app_config")).joinpath("app_config.json")
    config_path.write_text(test_app_config.json())
    return config_path


@pytest.fixture(scope="session")
def test_worker_config(test_app_config_path: Path) -> AMQPWorkerConfig:
    return AMQPWorkerConfig(
        log_level="DEBUG", app_bootstrap_config_path=test_app_config_path
    )


@pytest.fixture(scope="session")
async def app_lifetime_deps(test_worker_config: AMQPWorkerConfig):
    worker_id = "test-worker-id"
    async with app.lifetime_dependencies(
        worker_config=test_worker_config, worker_id=worker_id
    ):
        yield
