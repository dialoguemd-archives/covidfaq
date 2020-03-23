from datetime import date, datetime
from typing import List

from fastapi import APIRouter, HTTPException, Request, Body
from pydantic import BaseModel
from requests import HTTPError
from structlog import get_logger

from .. import config

from elasticsearch import Elasticsearch

from ..search.search_index import query_question

router = APIRouter()
log = get_logger()


class Answers(BaseModel):
    answers: List[str]

class ElasticResults(BaseModel):
    doc_text: List[str]
    doc_url: str
    section_text: List[str]
    section_url: str

conf = config.get()

es = Elasticsearch(
    [{"host": conf.elastic_search_host, "port": 443}],
    use_ssl=True,
    verify_certs=True,
)


# @router.get("/answers/", response_model=Answers)
@router.get("/answers/")
def answers(request: Request, data=Body(dict())):

    log.debug('data',data=data)
    question = data['question']
    print(question)
    print(request)

    log.info("answers", question=question)

    language = request.headers.get('Accept-Language')

    res_doc_txt, res_sec_txt = query_question(es, question, language)

    # answer = Answers.parse_obj(res_sec_txt)

    return language
