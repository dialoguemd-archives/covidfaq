#!/usr/bin/env python

import argparse
import json
import logging
import timeit

from covidfaq.evaluating.model.fake_reranker import FakeReRanker

logger = logging.getLogger(__name__)


def evaluate(model_to_evaluate, test_data):
    model_to_evaluate.collect_answers(test_data['answers'])
    correct = 0
    total = 0
    start = timeit.timeit()
    for question, target in test_data['questions'].items():
        prediction = model_to_evaluate.answer_question(question)
        if target == prediction:
            correct += 1
        total += 1
    end = timeit.timeit()
    logger.info('accuracy: {} / time: {}'.format(correct / total, end - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data',
                        help='file containing the data for evaluation', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_to_evaluate = FakeReRanker()

    with open(args.test_data, 'r') as in_steam:
        test_data = json.load(in_steam)

    evaluate(model_to_evaluate, test_data)


if __name__ == '__main__':
    main()
