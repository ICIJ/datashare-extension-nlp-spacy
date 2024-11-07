#!/bin/bash
N_PROCESSING_WORKERS=
poetry run python -m icij_worker workers start -g PYTHON -n "${N_PROCESSING_WORKERS:-1}" spacy_worker.app.app
