# from unittest.mock import patch

from fastapi.testclient import TestClient

from covidfaq.main import app

client = TestClient(app)


def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"healthy": "sure am"}


def test_format_language_fr(lang_fr, question_fr):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_fr, question_fr) == "fr"


def test_format_language_en(lang_en, question_en):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_en, question_en) == "en"


def test_format_language_other(lang_other, question_en):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_other, question_en) == "en"


def test_format_language_none(question_fr):
    from covidfaq.routers.answers import format_language

    assert format_language(None, question_fr) == "fr"
