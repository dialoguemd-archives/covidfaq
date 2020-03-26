from functools import lru_cache

import torch
from transformers import BertModel, BertTokenizer

from train_retriever import BertEncoder, Retriver, RetriverTrainer


@lru_cache()
def load_model():
    ## Berts
    model_str = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_str)
    bert_question = BertModel.from_pretrained(model_str)
    bert_paragraph = BertModel.from_pretrained(model_str)
    max_question_len_global = 30
    max_paragraph_len_global = 512

    encoder_question = BertEncoder(bert_question, max_question_len_global)
    encoder_paragarph = BertEncoder(bert_paragraph, max_paragraph_len_global)
    ret = Retriver(encoder_question, encoder_paragarph, tokenizer)

    model = RetriverTrainer(ret)
    tmp = torch.load("__temp_weight_ddp_end.ckpt")
    model.load_state_dict(tmp["state_dict"])

    return model


def re_rank(question, sections, topk=1):

    model = load_model()
    ranked_sections = model.retriever.predict(question, sections)

    # s1 = "dialogue rules!"
    # s2 = "covid sucks!"
    # s3 = "We <3 chatbots"
    # question = "When will we be able to go for team beers?"
    # sections = [s1, s2, s3]

    return ranked_sections[topk]
