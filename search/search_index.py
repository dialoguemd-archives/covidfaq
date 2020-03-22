#!/usr/bin/env python
# coding: utf-8

## Before running the script, install and start elastic search server on localhost port 9200 (by default)
from elasticsearch import Elasticsearch
import pandas as pd
from tqdm.auto import tqdm
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

def detect_language(text):
    return nlp(text)._.language

def search_document_index(es, index, query, topk):
    res = es.search({'query': {
        'match': {
            "content": query
            }
        },
        'size': topk
    }, index=index)
    return res

def search_section_index(es, index, query, topk, title_boost = 2):
    res = es.search({'query': {
        'multi_match': {
            "query": query,
            "fields": ["section^{}".format(title_boost), "content"]
            }
        },
        'size': topk
    }, index=index)
    return res

if __name__ == "__main__":

    en_doc_index = 'en-covid-document-index'
    en_sec_index = 'en-covid-section-index'
    fr_doc_index = 'fr-covid-document-index'
    fr_sec_index = 'fr-covid-section-index'

    ## Connect to the elastic cluster

    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if not es.ping():
        raise ValueError("Connection failed, please start server at localhost:9200 (default)")

    covid_questions = pd.read_csv('../data/website_questions_21032020_13h00.csv')['question'].tolist()

    topk_doc = 1
    topk_sec = 1

    for q in tqdm(covid_questions):
        if not isinstance(q, str):
            continue

        print('Question: ', q, '\n')

        # search top K sections
        if detect_language(q) == 'en':
            print(
                search_document_index(es, en_doc_index, q, topk_doc),
            )
            print(
                search_section_index(es, en_sec_index, q, topk_sec),
            )
        else:
            print(
                search_document_index(es, fr_doc_index, q, topk_doc),
            )
            print(
                search_section_index(es, fr_sec_index, q, topk_sec),
            )
        print('-' *  50)