class ModelEvaluationInterface:

    def collect_answers(self, answers):
        raise ValueError('implement')

    def answer_question(self, question):
        raise ValueError('implement')
