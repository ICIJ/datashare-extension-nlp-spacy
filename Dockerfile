# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye AS worker-base
ENV HOME=/home/user
ENV ICIJ_WORKER_TYPE=amqp

RUN apt-get update && apt-get install -y build-essential curl

RUN curl -LsSf https://astral.sh/uv/0.6.7/install.sh | sh
ENV PATH="$HOME/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="$HOME/.cargo/bin:$PATH"

WORKDIR $HOME/src/app

FROM worker-base AS worker
ADD scripts  ./scripts/
ARG n_workers
ENV N_PROCESSING_WORKERS=$n_workers
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project
# Then copy code
ADD uv.lock pyproject.toml README.md ./
ADD datashare_spacy_worker/  ./datashare_spacy_worker/
# Then install the worker lib
RUN uv sync -v --frozen --no-editable
RUN rm -rf ~/.cache/pip $(uv cache dir)
ENTRYPOINT ["/home/user/src/app/scripts/worker_entrypoint.sh"]


FROM worker-base AS worker-transformers
ADD scripts  ./scripts/
ARG n_workers
ENV N_PROCESSING_WORKERS=$n_workers
RUN --mount=type=cache,target=~/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync -v --frozen --no-editable --no-install-project --extra transformers
# Then copy code
ADD uv.lock pyproject.toml README.md ./
ADD datashare_spacy_worker/  ./datashare_spacy_worker/
# Then install the worker lib
RUN uv sync -v --frozen --no-editable --extra transformers
RUN rm -rf ~/.cache/pip $(uv cache dir)
ENTRYPOINT ["/home/user/src/app/scripts/worker_entrypoint.sh"]
