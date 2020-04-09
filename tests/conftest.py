from pytest import fixture


@fixture
def lang_fr():
    return "fr-CA"


@fixture
def lang_en():
    return "en-US"


@fixture
def lang_other():
    return "wrong language"


@fixture
def question_fr():
    return "Dois-je aller travailler?"


@fixture
def question_en():
    return "Should I go to work?"


@fixture
def elastic_results():
    return {
        "doc_results": [
            {
                "doc_text": ["doc string 1", "doc string 2"],
                "doc_url": "document_url.com",
            },
            {
                "doc_text": ["doc string 3", "doc string 4"],
                "doc_url": "document_url2.com",
            },
        ],
        "sec_results": [
            {
                "sec_text": ["sec string 1", "sec string 2"],
                "sec_url": "section_url.com",
            },
            {
                "sec_text": ["sec string 3", "sec string 4"],
                "sec_url": "section_url2.com",
            },
        ],
    }


@fixture
def ranked_scores():
    return [0.5, 0.9]
