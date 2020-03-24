#!/usr/bin/env python
# coding: utf-8

import json

import pandas as pd
import spacy
import structlog
from elasticsearch import Elasticsearch
from spacy_langdetect import LanguageDetector
from tqdm.auto import tqdm

from .build_index import en_doc_index, en_sec_index, fr_doc_index, fr_sec_index

log = structlog.get_logger(__name__)

nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

## Config variables
################################

question_file = "covidfaq/data/train_set_covid.csv"

################################


def detect_language(text):
    return nlp(text)._.language["language"]


def search_document_index(es, index, query, topk):
    res = es.search({"query": {"match": {"content": query}}, "size": topk}, index=index)
    return res


def search_section_index(es, index, query, topk, title_boost=1):
    res = es.search(
        {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["section^{}".format(title_boost), "content"],
                }
            },
            "size": topk,
        },
        index=index,
    )
    return res


def query_question(es, q, lan=None):
    if not lan:
        lan = detect_language(q)

    topk_doc = 1
    topk_sec = 1

    def helper(docindex, secindex):
        res_doc_txt = None
        res_sec_txt = None
        res_doc = search_document_index(es, docindex, q, topk_doc)["hits"]["hits"]
        if len(res_doc):
            res_doc_txt = res_doc[0]["_source"]
        res_sec = search_section_index(es, secindex, q, topk_sec)["hits"]["hits"]
        if len(res_sec):
            res_sec_txt = res_sec[0]["_source"]
        return res_doc_txt, res_sec_txt

    def formatter(res_doc_txt, res_sec_txt):
        formatted_data = {}

        if res_sec_txt:
            formatted_data["sec_text"] = res_sec_txt.get("content")
            formatted_data["sec_url"] = res_sec_txt.get("url")
        if res_doc_txt:
            doc_text = []
            for topic in json.loads(res_doc_txt.get("content")).keys():
                doc_text.extend(
                    json.loads(res_doc_txt.get("content")).get(topic).get("plaintext")
                )
            formatted_data["doc_url"] = res_doc_txt.get("url")
            formatted_data["doc_text"] = res_doc_txt.get("doc_text")

        return formatted_data

    # TODO: decode in unicode

    if lan == "en":
        res_doc_txt, res_sec_txt = helper(en_doc_index, en_sec_index)
        return formatter(res_doc_txt, res_sec_txt)
    else:
        res_doc_txt, res_sec_txt = helper(fr_doc_index, fr_sec_index)
        return formatter(res_doc_txt, res_sec_txt)


if __name__ == "__main__":

    es = Elasticsearch(
        [{"host": "es-covidfaq.dev.dialoguecorp.com", "port": 443}],
        use_ssl=True,
        verify_certs=True,
    )
    if not es.ping():
        raise ValueError(
            "Connection failed, please start server at localhost:9200 (default)"
        )

    covid_questions = pd.read_csv(question_file)["question"].tolist()

    all_results = []

    for q in tqdm(covid_questions):
        if not isinstance(q, str):
            continue

        record = {}

        answer = query_question(es, q)

        res_doc = answer.get("doc_url")
        res_sec = answer.get("sec_url")

        log.info("question_parsed", question=q, res_doc=res_doc, res_sec=res_sec)
