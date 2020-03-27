from pydantic import BaseModel
from structlog import get_logger

from fastapi import APIRouter

router = APIRouter()
log = get_logger()


class Health(BaseModel):
    healthy: str


@router.get("/health", response_model=Health)
def get_health():
    return Health(healthy="sure am")
