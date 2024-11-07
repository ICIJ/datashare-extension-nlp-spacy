# syntax=docker/dockerfile:1
FROM python:3.11-slim-bullseye AS worker-base
ENV HOME=/home/user
ENV ICIJ_WORKER_TYPE=amqp

RUN apt-get update && apt-get install -y build-essential curl

ENV POETRY_HOME=$HOME/.local/share/pypoetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="$HOME/.cargo/bin:$PATH"

WORKDIR $HOME/src/app
ADD scripts  ./scripts/
ADD datashare-spacy-worker/  ./datashare-spacy-worker/
ADD data/models.json  ./datashare-spacy-worker/spacy_worker/data/
WORKDIR $HOME/src/app/datashare-spacy-worker

FROM worker-base AS worker
ARG n_workers
ENV N_PROCESSING_WORKERS $n_workers
RUN --mount=type=cache,target=~/.cache/pypoetry poetry install
RUN rm -rf ~/.cache/pip ~/.cache/pypoetry/cache ~/.cache/pypoetry/artifacts
ENTRYPOINT ["/home/user/src/app/scripts/worker_entrypoint.sh"]


FROM worker-base AS worker-transformers
ARG n_workers
ENV N_PROCESSING_WORKERS $n_workers
RUN --mount=type=cache,target=~/.cache/pypoetry poetry install -E transformers
RUN rm -rf ~/.cache/pip ~/.cache/pypoetry/cache ~/.cache/pypoetry/artifacts
ENTRYPOINT ["/home/user/src/app/scripts/worker_entrypoint.sh"]
