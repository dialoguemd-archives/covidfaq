from dialogue.logging.fastapi import LogMiddleware
from dialogue.tracing.fastapi import TraceMiddleware
from fastapi import FastAPI
from structlog import get_logger

from . import config, routers

app = FastAPI()
app.add_middleware(LogMiddleware)
app.add_middleware(TraceMiddleware)
app.include_router(routers.health.router)
app.include_router(routers.answers.router)


@app.on_event("startup")
def on_startup():
    conf = config.get()
    log = get_logger()
    log.info("launching", **conf.dict())
