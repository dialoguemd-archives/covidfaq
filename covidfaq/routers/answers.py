from typing import List, Optional

import numpy as np

from fastapi import APIRouter, Request
from pydantic import BaseModel
from structlog import get_logger

from ..search.search_index import query_question
from ..utils import BertModels, ElasticSearchClient, detect_language, get_scores

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
def answers(request: Request, question: str, topk_es: int = None):

    language = request.headers.get("Accept-Language")
    es = ElasticSearchClient().es_client
    dbert_rerank_en = BertModels().dbert_rerank_en
    dbert_rerank_fr = BertModels().dbert_rerank_fr

    formatted_language = format_language(language, question)

    if not topk_es:
        topk_es = 5

    elastic_results = query_question(
        es, question, topk_sec=topk_es, lan=formatted_language
    )

    log.info(
        "elastic_results",
        elastic_results=elastic_results,
        question=question,
        language=language,
    )

    answers = []

    if elastic_results:
        elastic_results_formatted = ElasticResults.parse_obj(elastic_results)
        if elastic_results_formatted.sec_results:

            list_of_sec_results = elastic_results_formatted.sec_results

            # rerank
            sections_texts = [
                ", ".join(section.sec_text) for section in list_of_sec_results
            ]

            log.info(
                "reranking",
                formatted_language=formatted_language,
                question=question,
                sections_texts=sections_texts,
            )

            if formatted_language == "en":
                scores = get_scores(dbert_rerank_en, question, sections_texts)

            else:
                scores = get_scores(dbert_rerank_fr, question, sections_texts)

            max_index = np.argmax(scores)

            answers = [sections_texts[max_index]]

            log.info(
                "rerank_scores",
                scores=scores,
                question=question,
                sections_texts=sections_texts,
                max_index=max_index,
                answers=answers,
            )

    return {"answers": answers}


def format_language(language, question):

    if language and "en" in language.lower():
        return "en"
    elif language and "fr" in language.lower():
        return "fr"
    else:
        return detect_language(question)
