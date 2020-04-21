import torch
import tqdm
import yaml
from bert_reranker.models.load_model import load_model
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
        self.indices = None  # to be set with collect_answers

    def collect_answers(self, answers):
        cached_answers = []
        indices = []
        for index, answer in tqdm.tqdm(answers.items()):
            cached = self.model.embed_paragraph(answer)
            cached_answers.append(cached.squeeze(0))
            indices.append(index)
        self.p_embs = torch.stack(cached_answers)
        self.indices = indices

    def answer_question(self, question):
        q_emb = self.model.embed_question(question)
        logits = torch.mm(q_emb, self.p_embs.transpose(1, 0)).squeeze(0)
        scores = torch.sigmoid(logits)
        index_of_highest = torch.argmax(scores)
        return self.indices[index_of_highest]
