#!/usr/bin/env python

import argparse
import json
import logging
import time

from tqdm import tqdm

from bert_reranker.data.data_loader import get_passages_by_source
from covidfaq.evaluating.model.cheating_model import CheatingModel
from covidfaq.evaluating.model.elastic_search_reranker import ElasticSearchReRanker
from covidfaq.evaluating.model.embedding_based_reranker import EmbeddingBasedReRanker
from covidfaq.evaluating.model.embedding_based_reranker_plus_ood_detector import (
    EmbeddingBasedReRankerPlusOODDetector,
)
from covidfaq.evaluating.model.es_topk import ElasticSearchTopK
from covidfaq.evaluating.model.fake_reranker import FakeReRanker
from covidfaq.evaluating.model.google_model import GoogleModel
from covidfaq.evaluating.model.lda import LDAReranker
from covidfaq.evaluating.model.lsa import LSA
from covidfaq.evaluating.model.tfidf import TFIDF

logger = logging.getLogger(__name__)


def evaluate(model_to_evaluate, test_data):
    logger.info(
        "data is composed by {} examples and {} passages".format(
            len(test_data["examples"]), len(test_data["passages"])
        )
    )
    a_start = time.time()
    source2passages, passage_id2source, passage_id2index = get_passages_by_source(
        test_data
    )
    no_ood_source2passages, _, _ = get_passages_by_source(test_data, keep_ood=False)
    model_to_evaluate.collect_answers(no_ood_source2passages)

    a_end = time.time()
    answer_time = a_end - a_start
    logger.info(
        "preparing {} answers: total time {:.2f} sec./ per-answer time {:.2f} sec.".format(
            len(test_data["passages"]),
            answer_time,
            answer_time / len(test_data["passages"]),
        )
    )

    correct = 0
    total = 0
    q_start = time.time()

    for example in tqdm(test_data["examples"]):
        question = example["question"]
        source = example["source"]
        prediction = model_to_evaluate.answer_question(question, source)
        index_of_correct_passage = passage_id2index[example["passage_id"]]
        if index_of_correct_passage == prediction:
            correct += 1
        total += 1
    q_end = time.time()

    question_time = q_end - q_start
    logger.info(
        "producing the answers for {} questions: total time {:.2f} sec. / "
        "per-question time {:.2f} sec.".format(
            len(test_data["examples"]),
            question_time,
            question_time / len(test_data["examples"]),
        )
    )

    logger.info(
        "correct {} over {} / accuracy: {:.2f}%".format(
            correct, total, (correct / total) * 100
        )
    )


def topk_eval(model_to_evaluate, test_data, k=5):
    model_to_evaluate.collect_answers(test_data["answers"])

    correct = 0
    total = 0
    for question, target in tqdm.tqdm(test_data["questions"].items(), leave=False):
        prediction = model_to_evaluate.topk(question, k=k)
        if target in prediction:
            correct += 1
        total += 1

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
        "--eval-method",
        default="standard",
        help='Whether to evaluate the model using the "standard" accuracy, or with the "topk" method. By default, uses "standard".',
    )
    parser.add_argument(
        "--config", help="(optional) config file to initialize the model"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.test_data, "r") as in_stream:
        test_data = json.load(in_stream)

    if args.model_type == "fake":
        model_to_evaluate = FakeReRanker()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "embedding_based_reranker":
        if args.config is None:
            raise ValueError("model embedding_based_reranker requires --config")
        model_to_evaluate = EmbeddingBasedReRanker(args.config)
    elif args.model_type == "embedding_based_reranker_plus_ood":
        if args.config is None:
            raise ValueError("model embedding_based_reranker requires --config")
        model_to_evaluate = EmbeddingBasedReRankerPlusOODDetector(args.config)
    elif args.model_type == "cheating_model":
        _, _, passage_id2index = get_passages_by_source(test_data)
        model_to_evaluate = CheatingModel(test_data, passage_id2index)
    elif args.model_type == "elastic_search":
        model_to_evaluate = ElasticSearchReRanker()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "google_model":
        model_to_evaluate = GoogleModel()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "lsa":
        model_to_evaluate = LSA()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "lda":
        model_to_evaluate = LDAReranker()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "es_topk":
        model_to_evaluate = ElasticSearchTopK()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    elif args.model_type == "tfidf":
        model_to_evaluate = TFIDF()
        raise NotImplementedError("Model needs to be updated to new evaluator format")
    else:
        raise ValueError("--model_type={} not supported".format(args.model_type))

    if args.eval_method == "standard":
        evaluate(model_to_evaluate, test_data)
    elif args.eval_method == "topk":
        if not hasattr(model_to_evaluate, "topk"):
            raise AttributeError(
                "{} does not have the method 'topk'. Please use a different evaluation method.".format(
                    args.model_type
                )
            )

        topk_eval(model_to_evaluate, test_data, k=1)
        topk_eval(model_to_evaluate, test_data, k=5)
        topk_eval(model_to_evaluate, test_data, k=10)
        topk_eval(model_to_evaluate, test_data, k=20)
        topk_eval(model_to_evaluate, test_data, k=50)
    else:
        raise ValueError("--eval-method={} not supported".format(args.eval_method))


if __name__ == "__main__":
    main()
