#!/usr/bin/env python
# coding: utf-8

## Before running the script, install and start elastic search server on localhost port 9200 (by default)
from elasticsearch import Elasticsearch
# Connect to the elastic cluster
es=Elasticsearch([{'host':'localhost','port':9200}])


## Create local index if not exist
# es.indices.delete(index='covid-document-index', ignore=[400, 404])
# es.indices.delete(index='covid-section-index', ignore=[400, 404])
es.indices.create(index='covid-document-index', ignore=400)
es.indices.create(index='covid-section-index', ignore=400)


## Load covid document and sections
import pandas as pd
from tqdm.auto import tqdm

covid_questions = pd.read_csv('../data/website_questions_21032020_13h00.csv')['question'].tolist()


from os import listdir
from os.path import isfile, join
import json

scrape_path = '../scrape/'
jsonfiles = [f for f in listdir(scrape_path) if isfile(join(scrape_path, f)) if '.json' in f and 'faq' not in f]

c_d = 0
c_s = 0

for i in tqdm(jsonfiles):
    with open(scrape_path + i, 'r', encoding='utf-8') as f:
        t = json.load(f)
        doc = {
            'content': json.dumps(t),
            'file_name': i
        }
        tmp = es.index('covid-document-index', doc, id=i)
        c_d += tmp['_shards']['successful']
        c = 1
        for sec in t:
            rec = {
                'section': sec,
                'content': t[sec],
                'file_name': i
            }
            tmp = es.index('covid-section-index', rec, id=i + '_section_' + str(c))
            c_s += tmp['_shards']['successful']
            c += 1

print('Inserted/Updated {} documents and {} sections'.format(
    c_d,
    c_s
))

es.indices.refresh(index="covid-document-index")
es.indices.refresh(index="covid-section-index")

print('Total {} documents and {} sections'.format(
    es.cat.count('covid-document-index', params={"format": "json"})[0]['count'],
    es.cat.count('covid-section-index', params={"format": "json"})[0]['count']
))


# topk = 3
#
# covid_section_answers = {}
# covid_document_answers = {}
# for i in tqdm(covid_questions):
#     if not isinstance(i, str):
#         continue
#
#     print('Question: ', i, '\n')
#
#     # search top K sections
#     res= es.search({'query':{
#             'match':{
#                 "content": i
#             }
#         }
#     },index='covid-section-index')
#
#     for j in range(topk):
#         if j < len(res['hits']['hits']):
#             print('# {} Section: '.format(j + 1), res['hits']['hits'][j]['_source'], '\n')
#         else:
#             print('no matching #{} section'.format(j))
#     # search top 1 documents
#     res= es.search({'query':{
#             'match':{
#                 "content": i
#             }
#         }
#     },index='covid-document-index')
#
#     if len(res['hits']['hits']) > 0:
#         print('# 1 Document sections: ', [i for i in json.loads(res['hits']['hits'][0]['_source']['content'])])
#         print('# 1 Document: ',
#               res['hits']['hits'][0]['_source']['content'],
#               '\n')
#     else:
#         print('no matching documents')
#     print('-' *  50)




