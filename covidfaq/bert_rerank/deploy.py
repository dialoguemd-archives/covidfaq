import json
import torch
from torch import nn
from train_retriever import BertEncoder, Retriver, RetriverTrainer
from transformers import (
    BertModel,
    BertTokenizer,
)


if __name__ == '__main__':

    ## Berts
    model_str = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_str)
    bert_question = BertModel.from_pretrained(model_str)
    bert_paragraph = BertModel.from_pretrained(model_str)
    max_question_len_global = 30
    max_paragraph_len_global = 512

    encoder_question = BertEncoder(bert_question, max_question_len_global)
    encoder_paragarph = BertEncoder(bert_paragraph, max_paragraph_len_global)
    ret = Retriver(encoder_question, encoder_paragarph, tokenizer)

    ret_trainee = RetriverTrainer(ret)
    tmp = torch.load('__temp_weight_ddp_end.ckpt')

    s1 = 'dialogue rules!'
    s2 = 'covid sucks!'
    s3 = 'We <3 chatbots'
    q = 'When will we be able to go for team beers?'
    ret_trainee.load_state_dict(tmp['state_dict'])
    predictions = ret_trainee.retriever.predict(q, [s1, s2, s3])
