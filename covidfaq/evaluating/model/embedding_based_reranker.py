import logging

import yaml

from bert_reranker.data.data_loader import get_passage_last_header
from bert_reranker.models.load_model import load_model
from bert_reranker.models.retriever_trainer import RetrieverTrainer
from transformers import AutoTokenizer
from yaml import load

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)

import torch


logger = logging.getLogger(__name__)


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
        self.source2embedded_passages = {}

    def collect_answers(self, source2passages):
        self.source2embedded_passages = {}
        for source, passages in source2passages.items():
            logger.info("encoding source {}".format(source))
            if passages:
                passages_content = [get_passage_last_header(p) for p in passages]
                embedded_passages = self.model.embed_paragrphs(
                    passages_content, progressbar=True
                )
                self.source2embedded_passages[source] = embedded_passages
            else:
                self.source2embedded_passages[source] = None

    def answer_question(self, question, source, already_embedded=False):
        if already_embedded:
            enc_question = question
        else:
            enc_question = self.model.embed_question(question)
        embedded_candidates = self.source2embedded_passages[source]
        result, _ = self.model.predict(
            enc_question,
            embedded_candidates,
            passages_already_embedded=True,
            question_already_embedded=True,
        )
        return int(result)
