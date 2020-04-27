import urllib.request
from functools import lru_cache

import boto3
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text  # noqa
from structlog import get_logger

log = get_logger()


class Clusterer:
    class __Clusterer:
        def __init__(self):
            self.model = "https://tfhub.dev/google/universal-sentence-encoder/4"
            self.data_file = "covidfaq/clustering/clusters_all_in_en.csv"
            self.embed = self.load_embed()
            self.data, self.data_matrix = self.create_embed_matrix()

        @lru_cache()
        def load_embed(self):
            return hub.load(self.model)

        @lru_cache()
        def create_embed_matrix(self):
            data = pd.read_csv(self.data_file)

            return data, self.embed(data.question.values)

        def get_cluster(self, question, lang="en"):
            # check if fr
            if lang != "en":
                question = translate_fr_to_en(question, get_translate_client())

            q_embed = self.embed([question])

            sim_matrix = np.inner(q_embed, self.data_matrix)

            cluster = self.data.cluster.values[np.argmax(sim_matrix[0])]

            return cluster

    instance = None

    def __init__(self):
        if not Clusterer.instance:
            Clusterer.instance = Clusterer.__Clusterer()
        else:
            Clusterer.instance.embed = Clusterer.instance.load_embed()
            Clusterer.instance.model = (
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            )
            Clusterer.instance.data_file = "covidfaq/clustering/clusters_all_in_en.csv"
            (
                Clusterer.instance.data,
                Clusterer.instance.data_matrix,
            ) = Clusterer.instance.create_embed_matrix()

    def __getattr__(self, name):
        return getattr(self.instance, name)


@lru_cache()
def get_translate_client():
    return boto3.client(service_name="translate")


def translate_fr_to_en(question, translate_client):
    result = translate_client.translate_text(
        Text=question, SourceLanguageCode="fr", TargetLanguageCode="en"
    )
    return result.get("TranslatedText")


@lru_cache()
def get_labels_file():
    return pd.read_csv("covidfaq/clustering/clusters_covid_labels.csv")


def get_answer_from_cluster(cluster, lang="en"):

    if cluster == "unclassified":
        return

    labels_file = get_labels_file()

    answer_file = labels_file[labels_file.Label == cluster].Answer.values

    log.info("answer_file", answer_file=answer_file, cluster=cluster, lang=lang)

    if answer_file and ".md" in answer_file[0]:
        # answer_file = "03_test-referral.md"  # FOR TEST, TO REMOVE

        answer_file = answer_file[0]

        if lang != "en":
            answer_file = answer_file.split(".md")[0] + ".fr.md"

        try:
            f = urllib.request.urlopen(
                "https://raw.githubusercontent.com/dialoguemd/covid-19/master/src/regions/ca/info/faq/"
                + answer_file
            )

            return f.read().decode("utf-8")

        except:
            log.debug(
                "error_getting_answer_file",
                cluster=cluster,
                lang=lang,
                answer_file=answer_file,
            )


if __name__ == "__main__":

    my_clusterer = Clusterer()

    cluster = my_clusterer.get_cluster("est-ce que je vais mourrir?", lang="fr")
    print(cluster)

    answer = get_answer_from_cluster(cluster, "fr")

    print(answer)
