FROM rasa/rasa:1.8.1-full

ENV \
  PYTHONFAULTHANDLER=TRUE \
  PYTHONUNBUFFERED=TRUE \
  PYTHONWARNINGS=ignore

COPY data /app/data
COPY domain.yml /app
COPY config.yml /app
COPY credentials.yml /app

RUN rasa train -d domain.yml --out models -c config.yml --quiet
