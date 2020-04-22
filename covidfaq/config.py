from functools import lru_cache

from pydantic import BaseSettings


class Config(BaseSettings):
    elastic_search_host: str = "faq-master"
    elastic_search_port: int = 9200
    elastic_search_use_ssl: bool = False
#     elastic_search_verify_certs: bool = True

    class Config:
        env_prefix = ""
        case_insensitive = True


@lru_cache()
def get() -> Config:
    return Config()
