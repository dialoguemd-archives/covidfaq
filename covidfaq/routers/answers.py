import json
from typing import List

from bert_reranker.data.data_loader import get_passages_by_source
from fastapi import APIRouter, Request
from pydantic import BaseModel
from structlog import get_logger

from ..clustering.cluster import Clusterer, get_answer_from_cluster
from ..evaluating.model.embedding_based_reranker_plus_ood_detector import (
    EmbeddingBasedReRankerPlusOODDetector,
)
from ..utils import detect_language

router = APIRouter()
log = get_logger()


class Answers(BaseModel):
    answers: List[str]


@router.get("/answers", response_model=Answers)
def answers(request: Request, question: str, topk_es: int = None):

    clusterer = Clusterer()

    ood_reranker = EmbeddingBasedReRankerPlusOODDetector(
        "covidfaq/bert_en_model/config.yaml"
    )
    with open(
        "covidfaq/bert_en_model/quebec_faq_en_cleaned_20200522.json", "r"
    ) as in_stream:
        test_data = json.load(in_stream)
    source2passages, passage_id2source, passage_id2index = get_passages_by_source(
        test_data
    )
    ood_reranker.collect_answers(source2passages)

    language = request.headers.get("Accept-Language")

    formatted_language = format_language(language, question)

    if format_language == "en":
        idx_tensor = ood_reranker.answer_question(
            question, "20200522_quebec_faq_en_cleaned_collection4"
        )
        if idx_tensor == -1:
            # we are out of distribution
            answer = []
            log.info(
                "ood_result", answer=answer, question=question, language=language,
            )
        else:
            idx = idx_tensor.item()
            answer_dict = source2passages["20200522_quebec_faq_en_cleaned_collection4"][
                idx
            ]
            answer = answer_dict.get("reference").get("section_content")
            answer = [answer]
            log.info(
                "bert_rerank_result",
                answer_dict=answer_dict,
                idx=idx,
                answer=answer,
                question=question,
                language=language,
            )

    else:
        cluster = clusterer.get_cluster(question, lang=formatted_language)
        answer = get_answer_from_cluster(cluster, formatted_language)

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
