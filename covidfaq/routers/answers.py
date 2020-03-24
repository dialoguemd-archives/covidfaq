from functools import lru_cache
from typing import List, Optional

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Request
from pydantic import BaseModel
from structlog import get_logger

from .. import config
from ..search.search_index import query_question

router = APIRouter()
log = get_logger()


class Answers(BaseModel):
    answers: List[str]


class DocResults(BaseModel):
    doc_text: Optional[List[str]]
    doc_url: Optional[str]


class SecResults(BaseModel):
    sec_text: Optional[List[str]]
    sec_url: Optional[str]


class ElasticResults(BaseModel):
    doc_results: Optional[List[DocResults]]
    sec_results: Optional[List[SecResults]]


@router.get("/answers", response_model=Answers)
def answers(request: Request, question: str):

    language = request.headers.get("Accept-Language")
    es = get_es_client()

    if language:
        formatted_language = format_language(language)
        elastic_results = query_question(
            es, question, topk_sec=1, topk_doc=1, lan=formatted_language
        )

    else:
        elastic_results = query_question(es, question, topk_sec=1, topk_doc=1)

    log.info(
        "elastic_results",
        elastic_results=elastic_results,
        question=question,
        language=language,
    )

    elastic_results_formatted = ElasticResults.parse_obj(elastic_results)

    answers = []

    if elastic_results_formatted:
        if elastic_results_formatted.sec_results:
            answers = SecResults.parse_obj(
                elastic_results_formatted.sec_results[0]
            ).sec_text

    return {"answers": answers}


def format_language(language):
    if "en" in language.lower():
        return "en"
    elif "fr" in language.lower():
        return "fr"


@lru_cache()
def get_es_client():
    conf = config.get()

    return Elasticsearch(
        [{"host": conf.elastic_search_host, "port": 443}],
        use_ssl=True,
        verify_certs=True,
    )


@lru_cache()
def load_re_rank_model_en():
    return


def re_rank(model, question, sections_list):
    return
