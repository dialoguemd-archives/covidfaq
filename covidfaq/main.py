from fastapi import FastAPI
from structlog import get_logger

from . import config, routers

app = FastAPI()
app.include_router(routers.health.router)
app.include_router(routers.answers.router)


@app.on_event("startup")
def on_startup():
    conf = config.get()
    log = get_logger()
    log.info("launching", **conf.dict())
