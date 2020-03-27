from unittest.mock import patch

from fastapi.testclient import TestClient

from covidfaq.main import app

client = TestClient(app)


def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"healthy": "sure am"}


def test_format_language_fr(lang_fr):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_fr) == "fr"


def test_format_language_en(lang_en):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_en) == "en"


def test_format_language_other(lang_other):
    from covidfaq.routers.answers import format_language

    assert format_language(lang_other) is None


def test_answers(elastic_results):

    with patch(
        "covidfaq.routers.answers.get_es_client", return_value=None,
    ):
        with patch(
            "covidfaq.routers.answers.query_question", return_value=elastic_results,
        ):
            response = client.get(
                "/answers",
                params={"question": "Dois-je aller travailler?"},
                headers={"Accept-Language": "fr-CA"},
            )

    assert response.status_code == 200
    assert response.json() == {"answers": ["sec string 1", "sec string 2"]}


def test_answers_no_language(elastic_results):

    with patch(
        "covidfaq.routers.answers.get_es_client", return_value=None,
    ):
        with patch(
            "covidfaq.routers.answers.query_question", return_value=elastic_results,
        ):
            response = client.get(
                "/answers", params={"question": "Dois-je aller travailler?"},
            )

    assert response.status_code == 200
    assert response.json() == {"answers": ["sec string 1", "sec string 2"]}


def test_answers_no_results(elastic_results):

    with patch(
        "covidfaq.routers.answers.get_es_client", return_value=None,
    ):
        with patch(
            "covidfaq.routers.answers.query_question", return_value={},
        ):
            response = client.get(
                "/answers",
                params={"question": "Dois-je aller travailler?"},
                headers={"Accept-Language": "fr-CA"},
            )

    assert response.status_code == 200
    assert response.json() == {"answers": []}
