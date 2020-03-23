#!/usr/bin/env python
# coding: utf-8

import json
from os import listdir
from os.path import isfile, join

import structlog

## Before running the script, install and start elastic search server on localhost port 9200 (by default)
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

log = structlog.get_logger(__name__)


## Config variables
################################

scrape_path = "../scrape/"
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
    for i in tqdm(files):
        with open(i, "r", encoding="utf8") as f:
            t = json.load(f)
            t_ = {
                j: {k: t[j][k] for k in t[j] if k in ["plaintext", "URL"]}
                for j in t
                if j != doc_url_key
            }
            doc = {"content": json.dumps(t_), "file_name": i, "url": t[doc_url_key]}
            es.index(docindex, doc, id=i.split('/')[-1])
            c_d += 1
            c = 1
            for sec in t:
                if sec == doc_url_key:
                    continue
                rec = {
                    "section": sec,
                    "content": t[sec]["plaintext"],
                    "file_name": i,
                    "url": t[sec]["URL"],
                }
                es.index(secindex, rec, id=i.split('/')[-1] + "_section_" + str(c))
                c_s += 1
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

    ## Register index
    delete_index(es)
    create_index(es)

    ## Load files
    jsonfiles = [
        f
        for f in listdir(scrape_path)
        if isfile(join(scrape_path, f))
        if ".json" in f and "faq" not in f and "mainpage" not in f
    ]
    enfiles = [scrape_path + f for f in jsonfiles if "_en.json" in f]
    frfiles = [scrape_path + f for f in jsonfiles if "_fr.json" in f]

    fill_index(es, enfiles, en_doc_index, en_sec_index)
    fill_index(es, frfiles, fr_doc_index, fr_sec_index)