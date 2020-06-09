from functools import lru_cache

from pydantic import BaseSettings


class Config(BaseSettings):
    class Config:
        env_prefix = ""
        case_insensitive = True


@lru_cache()
def get() -> Config:
    return Config()
