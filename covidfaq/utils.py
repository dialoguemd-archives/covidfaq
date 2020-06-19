import spacy

from spacy_langdetect import LanguageDetector


nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)


def detect_language(text):
    return nlp(text)._.language["language"]
