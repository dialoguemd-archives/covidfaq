from unittest.mock import patch

from covidfaq.main import app
from fastapi.testclient import TestClient

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


def test_answers(elastic_results, ranked_scores):
    with patch("covidfaq.utils.get_es_client", return_value=None):
        with patch("covidfaq.utils.load_bert_models", return_value=(None, None)):
            with patch(
                "covidfaq.routers.answers.query_question", return_value=elastic_results,
            ):
                with patch(
                    "covidfaq.routers.answers.get_scores", return_value=ranked_scores
                ):
                    response = client.get(
                        "/answers",
                        params={"question": "Dois-je aller travailler?"},
                        headers={"Accept-Language": "fr-CA"},
                    )

    assert response.status_code == 200
    assert response.json() == {"answers": ["sec string 3, sec string 4"]}


def test_answers_no_language(elastic_results, ranked_scores):
    with patch("covidfaq.utils.get_es_client", return_value=None):
        with patch("covidfaq.utils.load_bert_models", return_value=(None, None)):
            with patch(
                "covidfaq.routers.answers.query_question", return_value=elastic_results,
            ):
                with patch(
                    "covidfaq.routers.answers.get_scores", return_value=ranked_scores
                ):
                    response = client.get(
                        "/answers", params={"question": "Dois-je aller travailler?"},
                    )

    assert response.status_code == 200
    assert response.json() == {"answers": ["sec string 3, sec string 4"]}


def test_answers_no_results():
    with patch("covidfaq.utils.get_es_client", return_value=None):
        with patch("covidfaq.utils.load_bert_models", return_value=(None, None)):
            with patch("covidfaq.routers.answers.query_question", return_value={}):
                response = client.get(
                    "/answers",
                    params={"question": "Dois-je aller travailler?"},
                    headers={"Accept-Language": "fr-CA"},
                )

    assert response.status_code == 200
    assert response.json() == {"answers": []}


def test_answers_no_section_results():
    with patch("covidfaq.utils.get_es_client", return_value=None):
        with patch("covidfaq.utils.load_bert_models", return_value=(None, None)):
            with patch(
                "covidfaq.routers.answers.query_question",
                return_value={
                    "doc_results": [
                        {
                            "doc_text": ["doc string 1", "doc string 2"],
                            "doc_url": "document_url.com",
                        },
                        {
                            "doc_text": ["doc string 3", "doc string 4"],
                            "doc_url": "document_url2.com",
                        },
                    ]
                },
            ):
                response = client.get(
                    "/answers",
                    params={"question": "Dois-je aller travailler?"},
                    headers={"Accept-Language": "fr-CA"},
                )

    assert response.status_code == 200
    assert response.json() == {"answers": []}
