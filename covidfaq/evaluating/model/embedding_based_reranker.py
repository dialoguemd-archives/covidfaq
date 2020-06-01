import logging

import torch
import yaml
from tqdm import tqdm

from bert_reranker.data.data_loader import _encode_passages, get_passage_last_header
from bert_reranker.models.load_model import load_model
from bert_reranker.models.retriever_trainer import RetrieverTrainer
from transformers import AutoTokenizer
from yaml import load

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)


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
        self.source2passages = source2passages
        source2encoded_passages, _, _ = _encode_passages(
            source2passages,
            self.ret_trainee.retriever.max_question_len,
            self.ret_trainee.retriever.tokenizer,
        )
        self.source2embedded_passages = {}
        for source, passages in source2passages.items():
            logger.info(
                "caching {} entries for source {}".format(len(passages), source)
            )
            embedded_passages = []
            for passage in tqdm(passages):
                passage_question = get_passage_last_header(passage)
                embedded_passage = self.model.embed_paragraph([passage_question])
                embedded_passages.append(embedded_passage.squeeze(0))
            self.source2embedded_passages[source] = torch.stack(embedded_passages).T

    def answer_question(self, question, source, already_embedded=False):
        if already_embedded:
            enc_question = question
        else:
            enc_question = self.model.embed_question(question)
        scores = torch.mm(enc_question, self.source2embedded_passages[source])
        result = torch.argmax(scores)
        return int(result)
