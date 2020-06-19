import os

import boto3
import structlog

from covidfaq.scrape import scrape
from covidfaq.k8s import rollout_restart
from covidfaq.evaluating.model.bert_plus_ood import BertPlusOOD

log = structlog.get_logger()


def upload_ood_to_s3():
    client = boto3.client("s3")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    file_to_upload = ("covidfaq/bert_en_model/ood_model.pkl",)
    client.upload_file(file_to_upload, BUCKET_NAME, "ood_model.pkl")
    log.info("OOD model uploaded to s3 bucket")


def instantiate_OOD():
    # instantiating the model will train and save the OOD detector
    BertPlusOOD()
    upload_ood_to_s3()


if __name__ == "__main__":
    scrape.run("covidfaq/scrape/quebec-sites.yaml", "covidfaq/scrape/")
    # This will train the OOD detector
    BertPlusOOD()
    upload_ood_to_s3()
    rollout_restart()
