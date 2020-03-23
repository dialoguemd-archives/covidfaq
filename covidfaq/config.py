from functools import lru_cache

from pydantic import BaseSettings, SecretStr


class Config(BaseSettings):
    elastic_search_host: str = "es-covidfaq.dev.dialoguecorp.com"

    class Config:
        env_prefix = ""
        case_insensitive = True


@lru_cache()
def get() -> Config:
    return Config()
