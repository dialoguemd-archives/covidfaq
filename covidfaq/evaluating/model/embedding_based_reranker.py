import torch
import tqdm
import yaml
from bert_reranker.main import init_model
from bert_reranker.models.load_model import load_model
from bert_reranker.models.pl_model_loader import try_to_restore_model_weights
from transformers import AutoTokenizer
from yaml import load

from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class EmbeddingBasedReRanker(ModelEvaluationInterface):

    def __init__(self, config):
        with open(config, 'r') as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)

        saved_model = hyper_params['saved_model']
        tokenizer_name = hyper_params['tokenizer_name']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = load_model(hyper_params, tokenizer, False)
        self.model.load_state_dict(torch.load(saved_model))

        self.p_embs = None  # to be set with collect_answers

    def collect_answers(self, answers):
        cached_answers = []
        for answer in tqdm.tqdm(answers):
            cached = self.model.embed_paragraph(answer)
            cached_answers.append(cached.squeeze(0))
        self.p_embs = torch.stack(cached_answers)

    def answer_question(self, question):
        q_emb = self.model.embed_question(question)

        logits = torch.mm(q_emb, self.p_embs.transpose(1, 0)).squeeze(0)
        return torch.argmax(logits)
