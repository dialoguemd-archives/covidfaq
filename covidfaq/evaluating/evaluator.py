#!/usr/bin/env python

import argparse
import json
import logging
import time

import tqdm

from covidfaq.evaluating.model.cheating_model import CheatingModel
from covidfaq.evaluating.model.embedding_based_reranker import EmbeddingBasedReRanker
from covidfaq.evaluating.model.fake_reranker import FakeReRanker

logger = logging.getLogger(__name__)


def evaluate(model_to_evaluate, test_data):
    logger.info('data is composed by {} questions and {} answers'.format(
        len(test_data['questions']), len(test_data['answers'])))
    a_start = time.time()
    model_to_evaluate.collect_answers(test_data['answers'])
    a_end = time.time()
    answer_time = a_end - a_start
    logger.info(
        'preparing {} answers: total time {:.2f} sec./ per-answer time {:.2f} sec.'.format(
            len(test_data['answers']), answer_time, answer_time / len(test_data['answers'])))

    correct = 0
    total = 0
    q_start = time.time()
    for question, target in tqdm.tqdm(test_data['questions'].items()):
        prediction = model_to_evaluate.answer_question(question)
        if target == prediction:
            correct += 1
        total += 1
    q_end = time.time()

    question_time = q_end - q_start
    logger.info(
        'producing the answers for {} questions: total time {:.2f} sec. / '
        'per-question time {:.2f} sec.'.format(
            len(test_data['questions']), question_time,
            question_time / len(test_data['questions'])))

    logger.info('correct {} over {} / accuracy: {:.2f}%'.format(
        correct, total, (correct / total) * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data',
                        help='file containing the data for evaluation', required=True)
    parser.add_argument('--model-type', help='model to evaluate', required=True)
    parser.add_argument('--config', help='(optional) config file to initialize the model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.test_data, 'r') as in_stream:
        test_data = json.load(in_stream)

    if args.model_type == 'fake':
        model_to_evaluate = FakeReRanker()
    elif args.model_type == 'embedding_based_reranker':
        if args.config is None:
            raise ValueError('model embedding_based_reranker requires --config')
        model_to_evaluate = EmbeddingBasedReRanker(args.config)
    elif args.model_type == 'cheating_model':
        model_to_evaluate = CheatingModel(test_data)
    else:
        raise ValueError('--model_type={} not supported'.format(args.model_type))

    evaluate(model_to_evaluate, test_data)


if __name__ == '__main__':
    main()
