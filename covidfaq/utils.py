from functools import lru_cache

from elasticsearch import Elasticsearch
from transformers import BertModel, BertTokenizer

from . import config


class ElasticSearchClient:
    class __ElasticSearchClient:
        def __init__(self):
            self.es_client = get_es_client()

    instance = None

    def __init__(self):
        if not ElasticSearchClient.instance:
            ElasticSearchClient.instance = ElasticSearchClient.__ElasticSearchClient()
        else:
            ElasticSearchClient.instance.es_client = get_es_client()

    def __getattr__(self, name):
        return getattr(self.instance, name)


class BertModels:
    class __BertModels:
        def __init__(self):
            self.tokenizer, self.bert_question, self.bert_paragraph = load_bert_models()

    instance = None

    def __init__(self):
        if not BertModels.instance:
            BertModels.instance = BertModels.__BertModels()
        else:
            BertModels.instance.tokenizer = load_bert_models()[0]
            BertModels.instance.bert_question = load_bert_models()[1]
            BertModels.instance.bert_paragraph = load_bert_models()[2]

    def __getattr__(self, name):
        return getattr(self.instance, name)


@lru_cache()
def get_es_client():
    conf = config.get()

    return Elasticsearch(
        [{"host": conf.elastic_search_host, "port": conf.elastic_search_port}],
    )


@lru_cache()
def load_bert_models():
    tokenizer = BertTokenizer.from_pretrained("covidfaq/rerank/cached_tokenizer/")
    bert_question = BertModel.from_pretrained("covidfaq/rerank/cached_bert/")
    bert_paragraph = BertModel.from_pretrained("covidfaq/rerank/cached_bert/")

    return tokenizer, bert_question, bert_paragraph
