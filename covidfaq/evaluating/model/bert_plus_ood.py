import json

from structlog import get_logger

from bert_reranker.data.data_loader import get_passages_by_source
from covidfaq.evaluating.model.embedding_based_reranker_plus_ood_detector import (
    EmbeddingBasedReRankerPlusOODDetector,
)

log = get_logger()

SOURCE = "quebec-faq"


class BertPlusOODEn:
    class __BertPlusOODEn:
        def __init__(self):
            self.model = EmbeddingBasedReRankerPlusOODDetector(
                "covidfaq/bert_en_model/config.yaml", lang="en"
            )

            test_data, _ = get_latest_scrape(lang="en")

            (
                self.source2passages,
                self.passage_id2source,
                self.passage_id2index,
            ) = get_passages_by_source(test_data, keep_ood=False)
            self.model.collect_answers(self.source2passages)

            self.get_answer("what are the symptoms of covid-19")

        def get_answer(self, question):
            idx = self.model.answer_question(question, SOURCE)
            if idx == -1:
                # we are out of distribution
                answer_complete = []
                log.info(
                    "bert_get_answer_ood", question=question, idx=idx,
                )
            else:
                answer_dict = self.source2passages[SOURCE][idx]
                section_header = answer_dict.get("reference").get("section_headers")
                answer = answer_dict.get("reference").get("section_converted_html")

                # HOTFIX: relative links in answers from qc website
                answer = fix_broken_links(answer)

                answer_complete = ["## " + section_header[0] + " \n\n " + answer]

                log.info(
                    "bert_get_answer",
                    question=question,
                    idx=idx,
                    section_header=section_header,
                    answer=answer,
                    answer_complete=answer_complete,
                )

            return answer_complete

    instance = None

    def __init__(self):
        if not BertPlusOODEn.instance:
            BertPlusOODEn.instance = BertPlusOODEn.__BertPlusOODEn()

    def __getattr__(self, name):
        return getattr(self.instance, name)


class BertPlusOODFr:
    class __BertPlusOODFr:
        def __init__(self):
            self.model = EmbeddingBasedReRankerPlusOODDetector(
                "covidfaq/bert_fr_model/config.yaml", lang="fr"
            )

            test_data, _ = get_latest_scrape(lang="fr")

            (
                self.source2passages,
                self.passage_id2source,
                self.passage_id2index,
            ) = get_passages_by_source(test_data, keep_ood=False)
            self.model.collect_answers(self.source2passages)

            self.get_answer("quels sont les symptomes de la covid")

        def get_answer(self, question):
            idx = self.model.answer_question(question, SOURCE)
            if idx == -1:
                # we are out of distribution
                answer_complete = []
                log.info(
                    "bert_get_answer_ood", question=question, idx=idx,
                )
            else:
                answer_dict = self.source2passages[SOURCE][idx]
                section_header = answer_dict.get("reference").get("section_headers")
                answer = answer_dict.get("reference").get("section_converted_html")
                answer_complete = ["## " + section_header[0] + " \n\n " + answer]

                log.info(
                    "bert_get_answer",
                    question=question,
                    idx=idx,
                    section_header=section_header,
                    answer=answer,
                    answer_complete=answer_complete,
                )

            return answer_complete

    instance = None

    def __init__(self):
        if not BertPlusOODFr.instance:
            BertPlusOODFr.instance = BertPlusOODFr.__BertPlusOODFr()

    def __getattr__(self, name):
        return getattr(self.instance, name)


def get_latest_scrape(lang="en"):

    latest_scrape_fname = "covidfaq/scrape/source_" + lang + "_faq_passages.json"

    with open(latest_scrape_fname) as in_stream:
        test_data = json.load(in_stream)

    return test_data, latest_scrape_fname


def fix_broken_links(answer):
    while answer is not None:
        to_be_returned = answer
        answer = fix_broken_link(answer)

    return to_be_returned


def fix_broken_link(answer):
    start = answer.find("](/")
    if start != -1:
        end = answer.find(")", answer.find("](/"))
        answer = answer.replace(
            answer[start + 2 : end], "quebec.ca" + answer[start + 2 : end]
        )
        link = answer[start + 2 : end + 9]
        new_link = remove_new_line_in_link(link)
        answer = answer.replace(link, new_link)
        return answer


def remove_new_line_in_link(link):
    return link.replace("\n", "")
