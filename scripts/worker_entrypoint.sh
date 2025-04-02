#!/bin/bash
N_PROCESSING_WORKERS=
uv run --no-sync icij-worker workers start -g Python -n "${N_PROCESSING_WORKERS:-1}" datashare_spacy_worker.app.app
