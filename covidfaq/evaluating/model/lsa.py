import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from covidfaq.evaluating.model.model_evaluation_interface import (
    ModelEvaluationInterface,
)


class LSA(ModelEvaluationInterface):
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
        self.vectorizer = TfidfVectorizer(
            max_df=0.9, min_df=1
        )

        self.svd = TruncatedSVD(300)

        answers = np.array(answers)

        self.answer_idx = np.array(answers[:, 0])
        answer_tfidf = self.vectorizer.fit_transform(answers[:, 1])
        self.answer_ls = self.svd.fit_transform(answer_tfidf)


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
        ls = self.svd.transform(enc)
        sim_scores = cosine_similarity(ls, self.answer_ls).squeeze()
        best_idx = self.answer_idx[sim_scores.argmax()]

        return int(best_idx)

    def topk(self, question, k=10):
        enc = self.vectorizer.transform([question])
        ls = self.svd.transform(enc)
        sim_scores = cosine_similarity(ls, self.answer_ls).squeeze()

        return sim_scores.argsort()[::-1][:k]
