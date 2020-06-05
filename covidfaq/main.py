from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from covidfaq import config, routers
from covidfaq.clustering.cluster import Clusterer
from covidfaq.scrape import scrape

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

    scrape.run("covidfaq/scrape/quebec-sites.yaml", "covidfaq/scrape/")
    log.info("done scraping")

    Clusterer()
