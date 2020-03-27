from fastapi import APIRouter
from pydantic import BaseModel
from structlog import get_logger

router = APIRouter()
log = get_logger()


class Health(BaseModel):
    healthy: str


@router.get("/health", response_model=Health)
def get_health():
    return Health(healthy="sure am")
