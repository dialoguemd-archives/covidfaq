from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel
from structlog import get_logger

from ..evaluating.model.bert_plus_ood import BertPlusOOD
from ..utils import detect_language

router = APIRouter()
log = get_logger()


class Answers(BaseModel):
    answers: List[str]


@router.get("/answers", response_model=Answers)
def answers(request: Request, question: str, topk_es: int = None):

    ood_reranker = BertPlusOOD()

    language = request.headers.get("Accept-Language")

    formatted_language = format_language(language, question)

    if formatted_language == "en":
        answers = ood_reranker.get_answer(question)
        log.info(
            "bert_rerank_result", answers=answers, question=question, language=language,
        )

    else:
        pass

        log.info(
            "clustering_results",
            cluster=cluster,
            answer=answer,
            question=question,
            language=language,
        )

        answers = []
        if answer:
            answers = [answer]

    return {"answers": answers}


def format_language(language, question):

    if language and "en" in language.lower():
        return "en"
    elif language and "fr" in language.lower():
        return "fr"
    else:
        return detect_language(question)
