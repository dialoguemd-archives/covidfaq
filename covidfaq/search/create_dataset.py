import pandas as pd
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch

from search_index import query_question
from covidfaq.routers.answers import SecResults, ElasticResults

## Config variables
################################

question_file = "/home/jerpint/covidfaq/covidfaq/data/covidquestions_07apr.csv"
#  question_file = "covidfaq/data/simple_2020-03-27T19_12_30.866993Z.csv"
#  question_file = "./simpleLsa_15n0.7dt_2020-03-27T19_12_30.866993Z.csv"
clustered_file = False
sample = True
use_local_es = True
################################



if __name__ == "__main__":

    topk_sec = 10
    topk_doc = 10

    if use_local_es:
        es = Elasticsearch(
            [{"host": "localhost", "port": 9200}],
        )

    else:
        es = Elasticsearch(
            [{"host": "es-covidfaq.dev.dialoguecorp.com", "port": 443}],
            use_ssl=True,
            verify_certs=True,
        )
    if not es.ping():
        raise ValueError(
            "Connection failed, please start server at localhost:9200 (default)"
        )

    covid_questions = pd.read_csv(question_file)

    if sample:
        if clustered_file:
            samples_per_cluster = 10
            covid_questions = covid_questions.groupby('cluster').apply(lambda x: x.sample(samples_per_cluster, random_state=42, replace=True)).reset_index(drop=True)
            covid_questions = covid_questions.drop(columns=['Unnamed: 0'])
        else:
            # sample for debugging
            covid_questions = covid_questions.iloc[0:100]

    else:
        if clustered_file:
            covid_questions = covid_questions.sort_values(by=['cluster']).reset_index(drop=True)
            covid_questions = covid_questions.drop(columns=['Unnamed: 0'])
            covid_questions = covid_questions.dropna(axis=0, subset=['cluster'])

    # Initialize rows for the different candidate answers:
    for ii in range(topk_sec):
        covid_questions['ans_' + str(ii)] = ''


    all_qa_pairs = []
    for row_idx, row in tqdm(covid_questions.iterrows()):
        question = row['question']
        if not isinstance(question, str):
            continue


        elastic_results = query_question(es, question, topk_sec=topk_sec, topk_doc=topk_doc)

        top_answers = []
        if elastic_results:
            elastic_results_formatted = ElasticResults.parse_obj(elastic_results)
            if elastic_results_formatted.sec_results:

               # List of all top answers, note that it can be less than topk_sec, but no more than topk_sec
               top_answers = [SecResults.parse_obj(elastic_results_formatted.sec_results[ii]).sec_text for ii in range(len(elastic_results_formatted.sec_results))]


        if top_answers:
            for idx, ans in enumerate(top_answers):
                covid_questions.at[row_idx, 'ans_' + str(idx)] = ans[0]

    if clustered_file:
        if sample:
            covid_questions.to_csv('sample_clusters_topk_responses_ES.csv', index=False)
        else:
            covid_questions.to_csv('clusters_topk_responses_ES.csv', index=False)
    else:
        if sample:
            covid_questions.to_csv('sample_questions_top10_responses_ES.csv', index=False)
        else:
            covid_questions.to_csv('all_questions_top10_responses_ES.csv', index=False)
