#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
from os import listdir
from os.path import isfile, join

from tqdm.auto import tqdm

import structlog
from elasticsearch import Elasticsearch

log = structlog.get_logger(__name__)


## Config variables
################################

scrape_path = "covidfaq/scrape/"
assert scrape_path[-1] == "/"

doc_url_key = "document_URL"

en_sec_index = "en-covid-section-index"
fr_sec_index = "fr-covid-section-index"

################################


def create_index(es):
    es.indices.create(index=en_sec_index, ignore=400)
    es.indices.create(index=fr_sec_index, ignore=400)


def delete_index(es):
    es.indices.delete(index=en_sec_index, ignore=[400, 404])
    es.indices.delete(index=fr_sec_index, ignore=[400, 404])


def fill_index(es, files, secindex):

    c_s = 0
    for file_ in tqdm(files):
        with open(file_, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            for idx, sec in enumerate(json_file):
                if sec == doc_url_key:
                    continue
                rec = {
                    "title": json_file[sec].get("title", sec),
                    "plaintext": json_file[sec].get("plaintext"),
                    "nested_title": json_file[sec].get("nested_title"),
                    "file_name": file_,
                    "url": json_file[sec]["url"],
                }
                tmp = es.index(secindex, rec, id=file_ + "_section_" + str(idx))
                c_s += tmp["_shards"]["successful"]
    log.info(
        "Inserted/Updated sections", secindex=secindex, c_s=c_s,
    )

    es.indices.refresh(index=secindex)

    log.info(
        "Total sections",
        secindex=secindex,
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

    fill_index(es, enfiles, en_sec_index)
    fill_index(es, frfiles, fr_sec_index)


if __name__ == "__main__":
    run()
