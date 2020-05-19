import logging

import structlog

from covidfaq.evaluating.model.elastic_search_reranker import ElasticSearchReRanker
from model import build_isolator

INDEX_NAME = "es_index"
logger = logging.getLogger(__name__)
log = structlog.get_logger(__name__)

# Currently unused, but might be useful
isolate_punc = build_isolator("~!@#$%^&*()_+`-={}|:<>?[]/;',.\"")


class ElasticSearchTopK(ElasticSearchReRanker):
    def __init__(self):
        super().__init__()

    def topk(self, question, k=10):

        res = self.es.search(
            {
                "query": {"multi_match": {"query": question, "fields": ["plaintext"],}},
                "size": k,
            },
            index=INDEX_NAME,
        )
        if res["hits"]["hits"]:
            return [x["_source"]["answer_index"] for x in res["hits"]["hits"]]
        else:
            return -1
