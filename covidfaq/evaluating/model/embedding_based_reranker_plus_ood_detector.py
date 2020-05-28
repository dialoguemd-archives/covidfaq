import pickle

import yaml
from yaml import load
import numpy as np

from covidfaq.evaluating.model.embedding_based_reranker import EmbeddingBasedReRanker


class EmbeddingBasedReRankerPlusOODDetector(EmbeddingBasedReRanker):

    def __init__(self, config):
        super(EmbeddingBasedReRankerPlusOODDetector, self).__init__(config)
        with open(config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
        outlier_model_pickle = hyper_params["outlier_model_pickle"]

        with open(outlier_model_pickle, 'rb') as file:
            outlier_detector_model = pickle.load(file)
        # predictor = PredictorWithOutlierDetector(self.ret_trainee, sklearn_model)
        self.outlier_detector_model = outlier_detector_model

    def collect_answers(self, source2passages):
        super(EmbeddingBasedReRankerPlusOODDetector, self).collect_answers(source2passages)

    def answer_question(self, question, source):

        emb_question = self.ret_trainee.retriever.embed_question(question)
        in_domain = self.outlier_detector_model.predict(emb_question)
        in_domain = np.squeeze(in_domain)
        if in_domain == 1:
            return super(EmbeddingBasedReRankerPlusOODDetector, self).answer_question(
                question, source)
        else:
            # for OOD, we return the index -1, and se set the score to 1.0
            return -1
