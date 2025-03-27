#!/bin/bash
N_PROCESSING_WORKERS=
poetry run python -m icij_worker workers start -g Python -n "${N_PROCESSING_WORKERS:-1}" datashare_spacy_worker.app.app
