FROM rasa/rasa:1.8.2-full

# install poetry
ARG POETRY_VERSION="1.0.0"
ENV \
  PYTHONFAULTHANDLER=TRUE \
  PYTHONUNBUFFERED=TRUE \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_NO_CACHE_DIR=off \
  PYTHONWARNINGS=ignore
RUN \
  pip install --upgrade pip \
  && pip install "poetry==${POETRY_VERSION}" \
  && poetry config virtualenvs.create false

# install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root -vvv

COPY data /app/data
COPY scrape /app/scrape
COPY search /app/search
COPY domain.yml /app
COPY config.fr.yml /app
COPY credentials.yml /app

# RUN sudo python -m spacy download fr_core_news_md
# RUN sudo python -m spacy link fr_core_news_md fr

# RUN rasa train --data data/fr/ -d domain.yml --out models/fr -c config.fr.yml --quiet

# RUN python scrape/scrape.py
# RUN python search/build_index.py
