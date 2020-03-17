FROM python:3.7.7-slim-buster as base

FROM base as builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++

ARG GEMFURY_TOKEN
ENV PIP_EXTRA_INDEX_URL https://pypi.fury.io/${GEMFURY_TOKEN}/dialogue/

# install poetry

ARG POETRY_VERSION="1.0.5"
ENV \
  PYTHONFAULTHANDLER=TRUE \
  PYTHONUNBUFFERED=TRUE \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_NO_CACHE_DIR=off
RUN \
  pip install --upgrade pip \
  && pip install "poetry==${POETRY_VERSION}" \
  && poetry config virtualenvs.create false

# install dependencies

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev --no-root -vvv

FROM base as final

# permit http service

ENV PORT 80
EXPOSE 80
WORKDIR /app

COPY --from=builder /usr/local/ /usr/local/
COPY . .
RUN rasa train -d data/domain.yml --out models -c policies.yml

ENTRYPOINT exec rasa run --enable-api --endpoints configs/endpoints.yml --connector rasa_stack.IOChannel --port ${PORT}
