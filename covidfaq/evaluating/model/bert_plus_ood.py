import json

from structlog import get_logger

from bert_reranker.data.data_loader import get_passages_by_source
from covidfaq.evaluating.model.embedding_based_reranker_plus_ood_detector import (
    EmbeddingBasedReRankerPlusOODDetector,
)

log = get_logger()

SOURCE = "quebec-faq"


class BertPlusOOD:
    class __BertPlusOOD:
        def __init__(self):
            self.model = EmbeddingBasedReRankerPlusOODDetector(
                "covidfaq/bert_en_model/config.yaml"
            )

            test_data, _ = get_latest_scrape()

            (
                self.source2passages,
                self.passage_id2source,
                self.passage_id2index,
            ) = get_passages_by_source(test_data, keep_ood=False)
            self.model.collect_answers(self.source2passages)

            self.get_answer("what are the symptoms of covid")

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
        if not BertPlusOOD.instance:
            BertPlusOOD.instance = BertPlusOOD.__BertPlusOOD()

    def __getattr__(self, name):
        return getattr(self.instance, name)


def get_latest_scrape():

    latest_scrape_fname = "covidfaq/scrape/source_en_faq_passages.json"

    with open(latest_scrape_fname) as in_stream:
        test_data = json.load(in_stream)

    return test_data, latest_scrape_fname
