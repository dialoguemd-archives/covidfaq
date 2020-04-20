import random

from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class CheatingModel(ModelEvaluationInterface):

    def __init__(self, gold_data):
        self.gold_data = gold_data

    def collect_answers(self, answers):
        pass

    def answer_question(self, question):
        return self.gold_data['questions'][question]
