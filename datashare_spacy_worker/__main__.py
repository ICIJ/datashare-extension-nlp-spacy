import os
import sys

from icij_worker import WorkerBackend
from icij_worker.backend import start_workers

from datashare_spacy_worker.app import PYTHON_TASK_GROUP

if __name__ == "__main__":
    # TODO: it would be nicer to have a dedicated utils to hit the top level CLI
    #  endpoint from Python main...
    # TODO: we should factorize this as much as possible with the DOCKER entrypoint,
    #  potentially the docker entry point should this main as entry point
    n_workers = os.environ.get("N_PROCESSING_WORKERS", 1)
    config_path = sys.argv[1]
    start_workers(
        "datashare_spacy_worker.app.app",
        n_workers,
        config_path,
        backend=WorkerBackend.MULTIPROCESSING,
        group=PYTHON_TASK_GROUP,
    )
