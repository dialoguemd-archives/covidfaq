import os
from zipfile import ZipFile

import boto3
import structlog

log = structlog.get_logger()


def fetch_model():
    rerank_dir = "covidfaq/rerank"
    model_name = f"{rerank_dir}/model.zip"

    log.info("downloading model from s3")
    s3 = boto3.client("s3")
    s3.download_file(
        "coviddata.dialoguecorp.com",
        "mirko/bert_rerank_model__for_testing.zip",
        model_name,
    )
    log.info("model downloaded")

    log.info("extracting model")
    with ZipFile(model_name, "r") as zip:
        zip.extractall(path=rerank_dir)
    log.info("model extracted")

    log.info("cleanup")
    os.remove(model_name)
    log.info("cleanup done")


def fetch_data():
    rerank_dir = "covidfaq/rerank"
    file_name = f"{rerank_dir}/covidfaq_data.zip"

    log.info("downloading covidfaq_data from s3")
    s3 = boto3.client("s3")
    s3.download_file(
        "coviddata.dialoguecorp.com",
        "mirko/bert_rerank_model__for_testing.zip",
        file_name,
    )
    log.info("model downloaded")

    log.info("extracting model")
    with ZipFile(file_name, "r") as zip:
        zip.extractall(path=rerank_dir)
    log.info("model extracted")

    log.info("cleanup")
    os.remove(file_name)
    log.info("cleanup done")


if __name__ == "__main__":
    fetch_model()
    fetch_data()
