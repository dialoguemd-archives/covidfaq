import logging

import structlog
from elasticsearch import Elasticsearch

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)

INDEX_NAME = "es_index"
logger = logging.getLogger(__name__)
log = structlog.get_logger(__name__)


class ElasticSearchReRanker(ModelEvaluationInterface):
    def __init__(self):
        # shut up annoying logging from elasti search
        logger = logging.getLogger("elasticsearch")
        logger.setLevel(logging.ERROR)

        es = Elasticsearch([{"host": "localhost", "port": "9200"}])
        if not es.ping():
            raise ValueError(
                "Connection failed, please start server at localhost:9200 (default)"
            )
        es.indices.delete(index=INDEX_NAME, ignore=[400, 404])
        es.indices.create(index=INDEX_NAME, ignore=400)
        self.es = es

    def collect_answers(self, answers):

        for answer_index, answer in answers:
            rec = {"plaintext": answer, "answer_index": answer_index}
            self.es.index(INDEX_NAME, rec)
        self.es.indices.refresh(index=INDEX_NAME)

    def answer_question(self, question):
        res = self.es.search(
            {
                "query": {"multi_match": {"query": question, "fields": ["plaintext"],}},
                "size": 1,
            },
            index=INDEX_NAME,
        )
        if res["hits"]["hits"]:
            return res["hits"]["hits"][0]["_source"]["answer_index"]
        else:
            return -1
