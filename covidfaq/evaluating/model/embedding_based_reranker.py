import torch
import tqdm
import yaml
from bert_reranker.models.load_model import load_model
from bert_reranker.models.retriever_trainer import RetrieverTrainer
from transformers import AutoTokenizer
from yaml import load

from covidfaq.evaluating.model.model_evaluation_interface import ModelEvaluationInterface


class EmbeddingBasedReRanker(ModelEvaluationInterface):

    def __init__(self, config):
        with open(config, 'r') as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)

        ckpt_to_resume = hyper_params['ckpt_to_resume']
        tokenizer_name = hyper_params['tokenizer_name']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = load_model(hyper_params, tokenizer, False)

        ret_trainee = RetrieverTrainer(model, None, None, None, None, None)

        model_ckpt = torch.load(
            ckpt_to_resume, map_location=torch.device("cpu")
        )
        ret_trainee.load_state_dict(model_ckpt["state_dict"])
        self.model = ret_trainee.retriever

        self.p_embs = None  # to be set with collect_answers
        self.indices = None  # to be set with collect_answers

    def collect_answers(self, answers):
        cached_answers = []
        indices = []
        for index, answer in tqdm.tqdm(answers.items()):
            cached = self.model.embed_paragraph(answer)
            cached_answers.append(cached.squeeze(0))
            indices.append(int(index))
        self.p_embs = torch.stack(cached_answers)
        self.indices = indices

    def answer_question(self, question):
        q_emb = self.model.embed_question(question)

        relevance_scores = torch.matmul(q_emb, self.p_embs.squeeze(0).T).squeeze(0)

        rerank_index = torch.argsort(-relevance_scores)
        index_of_highest = rerank_index[0].detach().cpu()

        return self.indices[index_of_highest]
