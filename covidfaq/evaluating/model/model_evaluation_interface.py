class ModelEvaluationInterface:

    def collect_answers(self, answers):
        """

        Parameters
        ----------
        answers: list[answer_index (type), answer (str)].

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

        Returns
        -------
        int: the correct answer_index (see collect_answers method) for this question.

        """
        raise ValueError('implement')
