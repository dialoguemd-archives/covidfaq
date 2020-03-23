FROM python:3.8.2-slim-buster as base

FROM base as builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++

# permit installing private packages

ARG GEMFURY_TOKEN
ENV PIP_EXTRA_INDEX_URL https://pypi.fury.io/${GEMFURY_TOKEN}/dialogue/

# install poetry

ARG POETRY_VERSION="1.0.5"
RUN \
  pip install --upgrade pip && \
  pip install "poetry==${POETRY_VERSION}" && \
  poetry config virtualenvs.create false

# standard python project

COPY pyproject.toml poetry.lock ./
RUN poetry install -vvv --no-dev

FROM base as final

# permit http service

ENV PORT 80
EXPOSE 80
WORKDIR /app

COPY --from=builder /usr/local /usr/local
RUN python -m spacy download en_core_web_md && \
    python -m spacy link en_core_web_md en
COPY . .

ENTRYPOINT exec hypercorn covidfaq.main:app --bind 0.0.0.0:${PORT}
