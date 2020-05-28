import yaml
from transformers import AutoTokenizer
from yaml import load

import torch
from bert_reranker.models.load_model import load_model
from bert_reranker.models.retriever_trainer import RetrieverTrainer
from bert_reranker.data.data_loader import _encode_passages
from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)


class EmbeddingBasedReRanker(ModelEvaluationInterface):
    """
    Model based on the BERT embedding approach.
    It will compute the embeddings for the various answers (and cache them).
    When a question is provided, it only needs to compute the embedding for that question.
    """

    def __init__(self, config):
        with open(config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)

        ckpt_to_resume = hyper_params["ckpt_to_resume"]
        tokenizer_name = hyper_params["tokenizer_name"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = load_model(hyper_params, tokenizer, False)

        self.ret_trainee = RetrieverTrainer(model, None, None, None, None, None)

        model_ckpt = torch.load(ckpt_to_resume, map_location=torch.device("cpu"))
        self.ret_trainee.load_state_dict(model_ckpt["state_dict"])
        self.model = self.ret_trainee.retriever

    def collect_answers(self, source2passages):
        self.source2passages = source2passages
        self.source2encoded_passages, _, _ = _encode_passages(
            source2passages, self.ret_trainee.retriever.max_question_len, self.ret_trainee.retriever.tokenizer)

    def answer_question(self, question, source):
        prediction, norm_score = self.ret_trainee.retriever.predict(
            question, self.source2encoded_passages[source])

        return prediction
