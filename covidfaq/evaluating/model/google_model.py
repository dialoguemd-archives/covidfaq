import json
import logging
import operator
import os

import requests

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)

logger = logging.getLogger(__name__)


class GoogleModel(ModelEvaluationInterface):
    def __init__(self):
        # Export your gcloud auth token to an env variable:
        # export GCLOUD_AUTH_TOKEN=$(gcloud auth application-default print-access-token)
        self.gcloud_auth_token = os.environ["GCLOUD_AUTH_TOKEN"]
        self.headers = {"Authorization": "Bearer " + self.gcloud_auth_token}
        self.url = "https://ml.googleapis.com/v1/projects/descartes-covid/models/hotline:predict?alt=json"

        self.failed_attempts = 0

    def collect_answers(self, answers):

        answers_list = []
        for ans in answers:
            answers_list.append(ans[1])

        self.data = {"instances": [{"candidates": answers_list}]}

    def answer_question(self, question):

        self.data["instances"][0]["input"] = question
        post_data = json.dumps(self.data)
        response = requests.post(self.url, data=post_data, headers=self.headers)
        if response.status_code == 200:
            predictions = response.json()

            scores_list = [
                pred["similarity_score"] for pred in predictions["predictions"]
            ]
            index, value = max(enumerate(scores_list), key=operator.itemgetter(1))

            return index
        else:
            self.failed_attempts += 1
            logger.info(
                "Something went wrong when making the request. Total wrong requests:  {}".format(
                    self.failed_attempts
                )
            )
            return -1
