from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class CheatingModel(ModelEvaluationInterface):
    """
    model that knows the golden truth and will always return the best result.
    (useful for debugging)
    """

    def __init__(self, gold_data):
        self.gold_data = gold_data

    def collect_answers(self, answers):
        pass

    def answer_question(self, question):
        return self.gold_data['questions'][question]
