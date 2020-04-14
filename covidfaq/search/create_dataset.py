import pandas as pd
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch

from search_index import query_question
from covidfaq.routers.answers import SecResults, ElasticResults

## Config variables
################################

#  question_file = "./covidquestions_07apr.csv"
question_file = "./simpleLsa_15n0.7dt_2020-03-27T19_12_30.866993Z.csv"
clustered_file = True  # Specify if the file contains clusters
sample = False  # Take only a subset of the questions (for debugging)
use_es = False  # Use ES to generate candidate answers
use_local_es = False  #  Only use this if you know how to set it up
topk = topk_sec = 10  # Top responses to fetch for ES
topk_doc = 10

################################



if __name__ == "__main__":

    # Read file
    covid_questions = pd.read_csv(question_file)

    # Remove punctuation and lowercase all the questions, keep original question
    covid_questions["question_processed"] = covid_questions['question'].str.replace('[^\w\s]','').str.lower()

    if sample:  # Generate only a subset (useful for debugging)
        if clustered_file:
            # Get 10 samples per cluster for representative cluster
            samples_per_cluster = 10
            covid_questions = covid_questions.groupby('cluster').apply(lambda x: x.sample(samples_per_cluster, random_state=42, replace=True)).reset_index(drop=True)
            covid_questions = covid_questions.drop(columns=['Unnamed: 0'])
        else:
            # sample for debugging
            covid_questions = covid_questions.iloc[0:100]

    else:
        if clustered_file: # Sort by cluster
            covid_questions = covid_questions.sort_values(by=['cluster']).reset_index(drop=True)
            covid_questions = covid_questions.drop(columns=['Unnamed: 0'])
            covid_questions = covid_questions.dropna(axis=0, subset=['cluster'])

            covid_questions = covid_questions[['timestamp_est', 'anonymous_id', 'question', 'question_processed', 'cluster']]
        else:
            covid_questions = covid_questions[['timestamp_est', 'anonymous_id', 'question', 'question_processed']]

    if use_es:
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


        # Initialize fields for the different topk candidate ES answers:
        for ii in range(topk_sec):
            covid_questions['ans_' + str(ii)] = ''


    if use_es:

        # Go through all questions, generate ES responses
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
                covid_questions.to_csv('sample_clusters_' + str(top_k) + '_responses_ES.csv', index=False)
            else:
                covid_questions.to_csv('clusters_' + str(topk) + '_responses_ES.csv', index=False)
        else:
            if sample:
                covid_questions.to_csv('sample_questions_' + str(topk) + '_responses_ES.csv', index=False)
            else:
                covid_questions.to_csv('all_questions_' + str(topk) + '_responses_ES.csv', index=False)

    else:

        covid_questions.to_csv('all_questions_clusters_sorted.csv', index=False)

        # Drop duplicates, keep only the processed question and cluster
        covid_questions_sub = covid_questions.drop_duplicates(subset='question_processed')[['question_processed', 'cluster']]
        covid_questions_sub.to_csv('question_and_clusters_only.csv', index=False)
