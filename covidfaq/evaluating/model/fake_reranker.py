from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class FakeReRanker(ModelEvaluationInterface):
    """
    Returns always the same answer.
    (use for debug)
    """

    def collect_answers(self, answers):
        pass

    def answer_question(self, question):
        return 1
