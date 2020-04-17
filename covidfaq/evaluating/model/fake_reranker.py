import random

from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class FakeReRanker(ModelEvaluationInterface):

    def collect_answers(self, answers):
        pass

    def answer_question(self, question):
        return 1
