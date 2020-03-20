FROM rasa/rasa:1.8.2-full

ENV \
  PYTHONFAULTHANDLER=TRUE \
  PYTHONUNBUFFERED=TRUE \
  PYTHONWARNINGS=ignore

COPY data /app/data
COPY domain.yml /app
COPY config.yml /app
COPY credentials.yml /app

RUN python -m spacy download fr_core_news_md
RUN python -m spacy link fr_core_news_md fr

RUN rasa train -d domain.yml --out models -c config.yml --quiet

# delete spacy files?
