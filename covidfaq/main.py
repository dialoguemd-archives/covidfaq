from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from covidfaq import config, routers

# from covidfaq.evaluating.model.bert_plus_ood import BertPlusOODEn, BertPlusOODFr
# from covidfaq.scrape.scrape import (
#     load_latest_source_data,
#     download_OOD_model,
#     download_cached_embeddings,
# )


app = FastAPI()
app.include_router(routers.health.router)
app.include_router(routers.answers.router)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def on_startup():
    conf = config.get()
    log = get_logger()
    log.info("launching", **conf.dict())

    # load_latest_source_data()
    # download_OOD_model()
    # download_cached_embeddings()
    # BertPlusOODEn()
    # BertPlusOODFr()
