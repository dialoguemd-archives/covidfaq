class ModelEvaluationInterface:

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
        raise ValueError('implement')

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
        raise ValueError('implement')
