class ModelEvaluationInterface:
    def collect_answers(self, source2passages):
        """

        Parameters
        ----------
        source2passages: {source (str): passages (list)}
        passages: list[passage (dict)]

        This is where the pre-processing (like caching) can happen.
        If your model does not cache, you can just save the answers (self.passages = passages)
        and use them in the answer_question method.

        Returns
        -------
        None

        """
        raise ValueError("implement")

    def answer_question(self, question, source):
        """
        Parameters
        ----------
        question: str,
        source: str, To be used in conjunction with source2passages to get all the passages from the given source.


        This method is called for every question. You will need to return the correct answer.
        The answers are provided to you by calling the method `collect_answers` above.

        Returns
        -------
        prediction: int, the correct answer_index (see collect_answers method's doc) for this question.

        """
        raise ValueError("implement")
