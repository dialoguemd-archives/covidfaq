import os

import boto3
import structlog

from covidfaq.scrape import scrape
from covidfaq.k8s import rollout_restart
from covidfaq.evaluating.model.bert_plus_ood import BertPlusOODEn, BertPlusOODFr

log = structlog.get_logger()


def upload_OOD_to_s3():
    client = boto3.client("s3")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    file_to_upload_en = "covidfaq/bert_en_model/en_ood_model.pkl"
    file_to_upload_fr = "covidfaq/bert_fr_model/fr_ood_model.pkl"
    client.upload_file(file_to_upload_en, BUCKET_NAME, "en_ood_model.pkl")
    client.upload_file(file_to_upload_fr, BUCKET_NAME, "fr_ood_model.pkl")
    log.info("OOD models uploaded to s3 bucket")


def upload_embeddings_to_s3():
    client = boto3.client("s3")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    file_to_upload_en = "covidfaq/bert_en_model/en_source2embedded_passages.tar"
    file_to_upload_fr = "covidfaq/bert_fr_model/fr_source2embedded_passages.tar"
    client.upload_file(
        file_to_upload_en, BUCKET_NAME, "en_source2embedded_passages.tar"
    )
    client.upload_file(
        file_to_upload_fr, BUCKET_NAME, "fr_source2embedded_passages.tar"
    )
    log.info("Passages embeddings uploaded to s3 bucket")


def instantiate_OOD():
    # instantiating the model will train and save the OOD detector
    scrape.load_latest_source_data()
    scrape.download_crowdsourced_data()
    BertPlusOODEn()
    BertPlusOODFr()
    upload_OOD_to_s3()
    upload_embeddings_to_s3()


if __name__ == "__main__":
    # Scrape the quebec sites, upload the results to aws
    scrape.run("covidfaq/scrape/quebec-sites.yaml", "covidfaq/scrape/")
    # This will train the OOD detector
    instantiate_OOD()
    rollout_restart()
