import pickle

import numpy as np
import yaml
from yaml import load

from covidfaq.evaluating.model.embedding_based_reranker import EmbeddingBasedReRanker
from covidfaq.evaluating.model.bert_plus_ood import get_latest_scrape


def train_OOD_detector(ret_trainee, json_file=None):
    '''
    Fit an OOD detector based on the the embeddings from the latest FAQ questions
    '''
    from bert_reranker.data.predict import generate_embeddings
    from bert_reranker.models.sklearn_outliers_model import fit_sklearn_model

    if not json_file:
        _, json_file = get_latest_scrape()
    embeddings_dict = generate_embeddings(ret_trainee, json_file, out_file='embeddings.npy')
    clf = fit_sklearn_model(embeddings_dict["passage_header_embs"],
                            model_name='local_outlier_factor',
                            output_filename='sklearn_model.pkl',
                            n_neighbors=4)
    return clf


class EmbeddingBasedReRankerPlusOODDetector(EmbeddingBasedReRanker):
    def __init__(self, config):
        super(EmbeddingBasedReRankerPlusOODDetector, self).__init__(config)
        with open(config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)

        # If a model is specified, load it, otherwise fit it on the new scrape
        if hyper_params["outlier_model_pickle"]:
            outlier_model_pickle = hyper_params["outlier_model_pickle"]

            with open(outlier_model_pickle, "rb") as file:
                outlier_detector_model = pickle.load(file)
            self.outlier_detector_model = outlier_detector_model
        else:
            self.outlier_detector_model = train_OOD_detector(self.ret_trainee)

    def collect_answers(self, source2passages):
        super(EmbeddingBasedReRankerPlusOODDetector, self).collect_answers(
            source2passages
        )

    def answer_question(self, question, source):
        emb_question = self.ret_trainee.retriever.embed_question(question)
        in_domain = self.outlier_detector_model.predict(emb_question)
        in_domain = np.squeeze(in_domain)
        if in_domain == 1:
            return super(EmbeddingBasedReRankerPlusOODDetector, self).answer_question(
                emb_question, source, already_embedded=True
            )
        else:
            # for OOD, we return the index -1, and se set the score to 1.0
            return -1
