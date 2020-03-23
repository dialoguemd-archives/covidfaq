from datetime import date, datetime
from typing import List

# import requests_async as requests
from fastapi import APIRouter, HTTPException
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


@router.get("/answers/", response_model=Answers)
def answers(question: str):
    conf = config.get()
    log.info("answers", question=question)

    es = Elasticsearch(
        [{"host": conf.elastic_search_host, "port": 443}],
        use_ssl=True,
        verify_certs=True,
    )

    ans = query_question(es, question)

    answer = Answers.parse_obj(ans)

    # fastapi how to parse header
    # Accept-Language

    return answer
