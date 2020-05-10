#!/usr/bin/env python

"""
Usage:
    cd covidfaq/covidfaq/evaluating
    poetry run python top_k_evaluator.py --test-data=faq_eval_data.json --model-type=es_enhanced
"""

import argparse
import json
import logging
import time

import tqdm

from covidfaq.evaluating.model.es_enhanced_reranker import ESEnhancedReRanker
from covidfaq.evaluating.model.lsa import LSA
from covidfaq.evaluating.model.lda import LDAReranker
from covidfaq.evaluating.model.tfidf import TFIDF

logger = logging.getLogger(__name__)


def topk_eval(model_to_evaluate, test_data, k=5):
    # logger.info(
    #     "data is composed by {} questions and {} answers".format(
    #         len(test_data["questions"]), len(test_data["answers"])
    #     )
    # )
    a_start = time.time()
    model_to_evaluate.collect_answers(test_data["answers"])
    a_end = time.time()
    answer_time = a_end - a_start
    # logger.info(
    #     "preparing {} answers: total time {:.2f} sec./ per-answer time {:.2f} sec.".format(
    #         len(test_data["answers"]),
    #         answer_time,
    #         answer_time / len(test_data["answers"]),
    #     )
    # )

    correct = 0
    total = 0
    q_start = time.time()
    for question, target in tqdm.tqdm(test_data["questions"].items(), leave=False):
        prediction = model_to_evaluate.topk(question, k=k)
        if target in prediction:
            correct += 1
        total += 1
    q_end = time.time()

    question_time = q_end - q_start
    # logger.info(
    #     "producing the answers for {} questions: total time {:.2f} sec. / "
    #     "per-question time {:.2f} sec.".format(
    #         len(test_data["questions"]),
    #         question_time,
    #         question_time / len(test_data["questions"]),
    #     )
    # )

    logger.info(
        "top-{} evaluation: correct {} over {} / accuracy: {:.2f}%".format(
            k, correct, total, (correct / total) * 100
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data", help="file containing the data for evaluation", required=True
    )
    parser.add_argument("--model-type", help="model to evaluate", required=True)
    parser.add_argument(
        "--config", help="(optional) config file to initialize the model"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.test_data, "r") as in_stream:
        test_data = json.load(in_stream)

    if args.model_type == "es_enhanced":
        model_to_evaluate = ESEnhancedReRanker()
    elif args.model_type == "lsa":
        model_to_evaluate = LSA()
    elif args.model_type == "lda":
        model_to_evaluate = LDAReranker()
    elif args.model_type == "tfidf":
        model_to_evaluate = TFIDF()
    else:
        raise ValueError(
            "--model_type={} not supported. Please make sure that it supports `model.topk` method calls".format(args.model_type)
        )

    topk_eval(model_to_evaluate, test_data, k=1)
    topk_eval(model_to_evaluate, test_data, k=5)
    topk_eval(model_to_evaluate, test_data, k=10)
    topk_eval(model_to_evaluate, test_data, k=20)
    topk_eval(model_to_evaluate, test_data, k=50)


if __name__ == "__main__":
    main()
