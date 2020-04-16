FROM rasa/rasa:1.9.3-full

ENV \
  PYTHONFAULTHANDLER=TRUE \
  PYTHONUNBUFFERED=TRUE \
  PYTHONWARNINGS=ignore

COPY covidfaq/data /app/data
COPY domain.yml /app
COPY config.fr.yml /app
COPY credentials.yml /app

# RUN sudo python -m spacy download fr_core_news_md
# RUN sudo python -m spacy link fr_core_news_md fr

RUN rasa train --data covidfaq/data/fr/ -d domain.yml --out models/fr -c config.fr.yml --quiet
