# from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface
#
#
# class ElasticsearchPlusreranker(ModelEvaluationInterface):
#
#     def __init__(self):
#         pass  # init ES db
#
#     def collect_answers(self, answers):
#         self.db.index(answers)
#
#     def answer_question(self, question):
#         candidates = self.db.retrieve(question)
#         final_result = self.bert_reranker.get_best(question, candidates)
#         return final_result
