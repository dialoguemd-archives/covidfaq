from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)


class CheatingModel(ModelEvaluationInterface):
    """
    model that knows the golden truth and will always return the best result.
    (useful for debugging)
    """

    def __init__(self, test_data, passage_id2index):
        self.test_data = test_data
        self.passage_id2index = passage_id2index

    def collect_answers(self, source2passages):
        pass

    def answer_question(self, question, source):

        for example in self.test_data['examples']:
            if question == example['question']:
                passage_id = example['passage_id']
                return self.passage_id2index[passage_id]
