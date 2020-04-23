import urllib.request
from functools import lru_cache

import boto3
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text  # noqa


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

    labels_file = get_labels_file()

    answer_file = labels_file[labels_file.Label == cluster].Answer.values

    if answer_file and ".md" in answer_file:
        answer_file = "03_test-referral.md"  # FOR TEST, TO REMOVE

        if lang != "en":
            answer_file = answer_file.split(".md")[0] + ".fr.md"

        f = urllib.request.urlopen(
            "https://raw.githubusercontent.com/dialoguemd/covid-19/master/src/regions/ca/info/faq/"
            + answer_file
        )

        return f.read().decode("utf-8")


if __name__ == "__main__":

    my_clusterer = Clusterer()

    cluster = my_clusterer.get_cluster("what are the symptoms?", lang="en")

    answer = get_answer_from_cluster(cluster, "en")

    print(cluster)
    print(answer)
