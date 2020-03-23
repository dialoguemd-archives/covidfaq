#!/usr/bin/env python
# coding: utf-8

import json
from copy import deepcopy

import pandas as pd
import spacy
## Before running the script, install and start elastic search server on localhost port 9200 (by default)
from elasticsearch import Elasticsearch
from spacy_langdetect import LanguageDetector
from tqdm.auto import tqdm

from build_index import en_doc_index, en_sec_index, fr_doc_index, fr_sec_index

nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

## Config variables
################################

question_file = "search/query_result_2020-03-22T22_05_26.81834Z.csv"

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


def query_question(es, q):
    lan = detect_language(q)

    topk_doc = 1
    topk_sec = 1

    def helper(docindex, secindex):
        res_doc = search_document_index(es, docindex, q, topk_doc)["hits"]["hits"]
        if len(res_doc):
            res_doc_txt = json.dumps(res_doc[0]["_source"])
        else:
            res_doc_txt = "None"
        res_sec = search_section_index(es, secindex, q, topk_sec)["hits"]["hits"]
        if len(res_sec):
            res_sec_txt = json.dumps(res_sec[0]["_source"])
        else:
            res_sec_txt = "None"
        return res_doc_txt, res_sec_txt

    if lan == "en":
        return helper(en_doc_index, en_sec_index)
    elif lan == "fr":
        return helper(fr_doc_index, fr_sec_index)
    else:
        return ("None", "None")


if __name__ == "__main__":

    ## Connect to the elastic cluster

    es = Elasticsearch([{"host": "localhost", "port": 9200}])
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

        print("$$ Question: ", q, "\n")

        res_doc, res_sec = query_question(es, q)

        print("=== Document search result: ", res_doc)
        print()
        print("=== Section search result: ", res_sec)
        print("-" * 50)

        record["question"] = q
        record["document_result_json"] = res_doc
        record["section_result_json"] = res_sec

        all_results.append(deepcopy(record))

    pd.DataFrame(
        all_results, columns=["question", "section_result_json", "document_result_json"]
    ).to_csv("search/all_results.csv")
