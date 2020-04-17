#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import spacy
import structlog
from elasticsearch import Elasticsearch
from spacy_langdetect import LanguageDetector
from tqdm.auto import tqdm

from covidfaq.search.build_index import (
    en_sec_index,
    fr_sec_index,
    get_es_hostname,
    get_es_port,
)

log = structlog.get_logger(__name__)

nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

## Config variables
################################

question_file = "covidfaq/data/train_set_covid.csv"

################################


def detect_language(text):
    return nlp(text)._.language["language"]


def search_section_index(es, index, query, topk):
    res = es.search(
        {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "plaintext", "nested_title"],
                }
            },
            "size": topk,
        },
        index=index,
    )
    return res


def helper(es, q, topk_sec, secindex):
    res_sec_txt = None
    res_sec = search_section_index(es, secindex, q, topk_sec)["hits"]["hits"]
    if len(res_sec):
        res_sec_txt = [sec["_source"] for sec in res_sec]
    return res_sec_txt


def formatter(res_sec_txt):
    formatted_data = {}
    if res_sec_txt:
        res_sec_list = []
        for sec in res_sec_txt:
            formatted_sec = {}
            formatted_sec["sec_text"] = sec.get("content")
            formatted_sec["sec_url"] = sec.get("url")
            res_sec_list.append(formatted_sec)
        formatted_data["sec_results"] = res_sec_list
    return formatted_data


def query_question(es, q, topk_sec=1, lan=None):
    if not lan:
        lan = detect_language(q)

    if lan == "en":
        res_sec_txt = helper(es, q, topk_sec, en_sec_index)
        return formatter(res_sec_txt)
    else:
        res_sec_txt = helper(es, q, topk_sec, fr_sec_index)
        return formatter(res_sec_txt)


if __name__ == "__main__":

    from covidfaq.routers.answers import SecResults, ElasticResults

    host = get_es_hostname()
    port = get_es_port()

    log.info("ElasticSearch server details: ", port=port, host=host)

    if host == "localhost":
        es = Elasticsearch([{"host": host, "port": port}],)
    else:
        es = Elasticsearch(
            [{"host": host, "port": port}], use_ssl=True, verify_certs=True,
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

        elastic_results = query_question(es, q)
        if elastic_results:
            elastic_results_formatted = ElasticResults.parse_obj(elastic_results)
            if elastic_results_formatted.sec_results:

                # List of all top answers, note that it can be less than topk_sec, but no more than topk_sec
                top_answers = [
                    SecResults.parse_obj(
                        elastic_results_formatted.sec_results[ii]
                    ).sec_text
                    for ii in range(len(elastic_results_formatted.sec_results))
                ]
                top_answer = top_answers[0]
