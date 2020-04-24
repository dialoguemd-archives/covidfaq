FROM python:3.7.6-slim-buster as base

FROM base as builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ wget gnupg2

# install google chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

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

ENV PORT 80
EXPOSE 80
WORKDIR /app

COPY --from=builder /usr/local /usr/local
RUN python -m spacy download en_core_web_md && \
    python -m spacy link en_core_web_md en

COPY covidfaq/rerank covidfaq/rerank

COPY . .

ENTRYPOINT exec hypercorn covidfaq.main:app --bind 0.0.0.0:${PORT}
