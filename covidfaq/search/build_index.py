#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
from os import listdir
from os.path import isfile, join

import structlog
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

log = structlog.get_logger(__name__)


## Config variables
################################

scrape_path = "covidfaq/scrape/"
assert scrape_path[-1] == "/"

doc_url_key = "document_URL"

en_doc_index = "en-covid-document-index"
en_sec_index = "en-covid-section-index"
fr_doc_index = "fr-covid-document-index"
fr_sec_index = "fr-covid-section-index"

################################


def create_index(es):
    es.indices.create(index=en_doc_index, ignore=400)
    es.indices.create(index=en_sec_index, ignore=400)
    es.indices.create(index=fr_doc_index, ignore=400)
    es.indices.create(index=fr_sec_index, ignore=400)


def delete_index(es):
    es.indices.delete(index=en_doc_index, ignore=[400, 404])
    es.indices.delete(index=en_sec_index, ignore=[400, 404])
    es.indices.delete(index=fr_doc_index, ignore=[400, 404])
    es.indices.delete(index=fr_sec_index, ignore=[400, 404])


def fill_index(es, files, docindex, secindex):

    c_d = 0
    c_s = 0
    for file_ in tqdm(files):
        with open(file_, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            content = {
                j: {k: json_file[j][k] for k in json_file[j] if k in ["plaintext"]}
                for j in json_file
                if j != doc_url_key
            }
            doc = {
                "content": json.dumps(content),
                "file_name": file_,
                "url": json_file[doc_url_key],
            }
            tmp = es.index(docindex, doc, id=file_)
            c_d += tmp["_shards"]["successful"]
            c = 1
            for sec in json_file:
                if sec == doc_url_key:
                    continue
                rec = {
                    "section": sec,
                    "content": json_file[sec]["plaintext"],
                    "file_name": file_,
                    "url": json_file[sec]["url"],
                }
                tmp = es.index(secindex, rec, id=file_ + "_section_" + str(c))
                c_s += tmp["_shards"]["successful"]
                c += 1
    log.info(
        "Inserted/Updated documents and sections",
        docindex=docindex,
        secindex=secindex,
        c_d=c_d,
        c_s=c_s,
    )

    es.indices.refresh(index=docindex)
    es.indices.refresh(index=secindex)

    log.info(
        "Total documents and sections",
        docindex=docindex,
        secindex=secindex,
        c_d=es.cat.count(docindex, params={"format": "json"})[0]["count"],
        c_s=es.cat.count(secindex, params={"format": "json"})[0]["count"],
    )


def get_es_hostname():
    return os.environ.get("elastic_search_host", "faq-master.covidfaq")


def get_es_port():
    return os.environ.get("elastic_search_port", 9200)


def run():

    es = Elasticsearch([{"host": get_es_hostname(), "port": get_es_port()}])
    if not es.ping():
        raise ValueError(
            "Connection failed, please start server at localhost:9200 (default)"
        )

    ## Register index
    delete_index(es)
    create_index(es)

    ## Load files
    jsonfiles = [
        f for f in listdir(scrape_path) if isfile(join(scrape_path, f)) if ".json" in f
    ]
    enfiles = [scrape_path + f for f in jsonfiles if re.search(r"-en-.*\.json$", f)]
    frfiles = [scrape_path + f for f in jsonfiles if re.search(r"-fr-.*\.json$", f)]

    fill_index(es, enfiles, en_doc_index, en_sec_index)
    fill_index(es, frfiles, fr_doc_index, fr_sec_index)


if __name__ == "__main__":
    run()
