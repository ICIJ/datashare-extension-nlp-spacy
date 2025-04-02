import multiprocessing
import os
from pathlib import Path
from typing import Annotated

import typer
from icij_worker import WorkerBackend
from icij_worker.backend import start_workers

from datashare_spacy_worker.app import PYTHON_TASK_GROUP

cli_app = typer.Typer()


@cli_app.command()
def main(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
    n_workers: Annotated[
        int, typer.Option("-n", "--n-workers", help="Number of NLP workers")
    ] = 1,
) -> None:
    os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
    start_workers(
        "datashare_spacy_worker.app.app",
        n_workers,
        config_path,
        backend=WorkerBackend.MULTIPROCESSING,
        group=PYTHON_TASK_GROUP,
    )


if __name__ == "__main__":
    # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    multiprocessing.freeze_support()
    cli_app()
