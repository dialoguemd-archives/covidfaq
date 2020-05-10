import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)

def build_isolator(chars):
    def isolate(text):
        for c in chars:
            text = text.replace(c, f" {c}")
        return text
    return isolate


class LDAReranker(ModelEvaluationInterface):
    def collect_answers(self, answers):
        """

        Parameters
        ----------
        answers: list[answer_index (int), answer (str)].

        This is where the pre-processing (like caching) can happen.
        If your model does not cache, you can just save the answers (self.answers = answers)
        and use them in the answer_question method.

        Returns
        -------

        """
        self.vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, stop_words='english'
        )
        
        self.lda = LatentDirichletAllocation(n_components=100)

        answers = np.array(answers)

        self.answer_idx = np.array(answers[:, 0])
        answer_tf = self.vectorizer.fit_transform(answers[:, 1])
        self.answer_lda = self.lda.fit_transform(answer_tf)


    def answer_question(self, question):
        """

        Parameters
        ----------
        question: str

        This method is called for every question. You will need to return the correct answer.
        The answers are provided to you by calling the method `collect_answers` above.

        Returns
        -------
        int: the correct answer_index (see collect_answers method's doc) for this question.

        """
        enc = self.vectorizer.transform([question])
        ls = self.lda.transform(enc)
        sim_scores = cosine_similarity(ls, self.answer_lda).squeeze()
        best_idx = self.answer_idx[sim_scores.argmax()]

        return int(best_idx)

    def topk(self, question, k=10):
        enc = self.vectorizer.transform([question])
        ls = self.lda.transform(enc)
        sim_scores = cosine_similarity(ls, self.answer_lda).squeeze()

        return sim_scores.argsort()[::-1][:k]
