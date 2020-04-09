import os
import pickle
from functools import lru_cache

import numpy as np
import spacy
import tensorflow as tf
import tensorflow.keras.layers as L
from elasticsearch import Elasticsearch
from spacy_langdetect import LanguageDetector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tokenizers import BertWordPieceTokenizer
from transformers import TFAutoModel, TFElectraModel

from . import config

nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)


class ElasticSearchClient:
    class __ElasticSearchClient:
        def __init__(self):
            self.es_client = get_es_client()

    instance = None

    def __init__(self):
        if not ElasticSearchClient.instance:
            ElasticSearchClient.instance = ElasticSearchClient.__ElasticSearchClient()
        else:
            ElasticSearchClient.instance.es_client = get_es_client()

    def __getattr__(self, name):
        return getattr(self.instance, name)


class BertModels:
    class __BertModels:
        def __init__(self):
            self.dbert_rerank_en, self.dbert_rerank_fr = load_bert_models()

    instance = None

    def __init__(self):
        if not BertModels.instance:
            BertModels.instance = BertModels.__BertModels()
        else:
            BertModels.instance.dbert_rerank_en = load_bert_models()[0]
            BertModels.instance.dbert_rerank_fr = load_bert_models()[1]

    def __getattr__(self, name):
        return getattr(self.instance, name)


def detect_language(text):
    return nlp(text)._.language["language"]


@lru_cache()
def get_es_client():
    conf = config.get()

    return Elasticsearch(
        [{"host": conf.elastic_search_host, "port": conf.elastic_search_port}],
    )


@lru_cache()
def load_bert_models():
    dbert_en = load_model(
        sigmoid_dir="covidfaq/rerank/model_dir_en/",
        transformer_dir="covidfaq/rerank/model_dir_en/transformer/",
        max_len=None,
    )

    dbert_fr = load_model(
        sigmoid_dir="covidfaq/rerank/model_dir_fr/",
        transformer_dir="covidfaq/rerank/model_dir_fr/transformer/",
        max_len=None,
    )

    dbert_tokenizer_en = BertWordPieceTokenizer(
        "covidfaq/rerank/model_dir_en/vocab.txt", lowercase=True
    )

    dbert_tokenizer_fr = BertWordPieceTokenizer(
        "covidfaq/rerank/model_dir_fr/vocab.txt", lowercase=True
    )

    dbert_rerank_en = build_reranker(dbert_tokenizer_en, dbert_en)

    dbert_rerank_fr = build_reranker(dbert_tokenizer_fr, dbert_fr)

    return dbert_rerank_en, dbert_rerank_fr


def build_reranker(tokenizer, model):
    tokenizer.enable_padding()

    def rerank(question, answers):
        pairs = list(zip([question] * len(answers), answers))

        encs = tokenizer.encode_batch(pairs)
        input_ids = np.array([enc.ids for enc in encs])
        scores = model.predict(input_ids[:512]).squeeze()

        return scores

    return rerank


def build_model(transformer, max_len=256):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_ids = L.Input(shape=(max_len,), dtype=tf.int32)

    x = transformer(input_ids)[0]
    x = x[:, 0, :]
    x = L.Dense(1, activation="sigmoid", name="sigmoid")(x)

    # BUILD AND COMPILE MODEL
    model = Model(inputs=input_ids, outputs=x)
    model.compile(
        loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(lr=1e-5)
    )

    return model


def load_model(
    sigmoid_dir, transformer_dir="transformer", architecture="distilbert", max_len=256
):
    """
    Special function to load a keras model that uses a transformer layer
    """
    sigmoid_path = os.path.join(sigmoid_dir, "sigmoid.pickle")

    if architecture == "electra":
        transformer = TFElectraModel.from_pretrained(transformer_dir)
    else:
        transformer = TFAutoModel.from_pretrained(transformer_dir)
    model = build_model(transformer, max_len=max_len)

    sigmoid = pickle.load(open(sigmoid_path, "rb"))
    model.get_layer("sigmoid").set_weights(sigmoid)

    return model


def get_scores(dbert_rerank, question, sections_texts):
    return dbert_rerank(question, sections_texts)
