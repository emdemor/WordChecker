from sys import stderr
from typing import Any, Dict
import spacy
import difflib
import numpy as np
import pandas as pd
from scipy.stats import hmean
from unidecode import unidecode
from wordfreq import get_frequency_dict


class WordChecker:
    """
    The WordChecker class is used to check the existence of
    a word in a dictionary of a certain language and find
    similar words.

    Parameters
    ----------
    lang : str
        Language chosen for the dictionary.

    Attributes
    ----------
    nlp : object
        Object from the spacy package with the information of the chosen language.
    frequency : dict
        Dictionary with the frequency of words in the chosen language.
    vocabulary : array
        List of words present in the dictionary of the chosen language.
    """

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.nlp = download_spacy_model(self.__get_spacy_model(self.lang))
        self.frequency = self.__set_language_frequency()
        self.vocabulary = self.__set_vocabulary()

    def __call__(self, word: str) -> bool:
        """Check if a word exists in the dictionary."""
        return word in self.vocabulary

    def most_similar_word(self, word: str) -> str:
        """Find the most similar word to the one passed as a parameter."""
        list_ = self.most_similar_n_words(word)
        return self.__rule_to_word_selection(list_)

    def correct_word(self, word: str, n: int = 3, cutoff: float = 0.6) -> np.ndarray:
        """Find the n most similar words to coorect the one passed as a parameter."""

        word = word.lower()

        filtered_vocab = self.vocabulary[
            [(word[0] == w[0]) and (word[-1] == w[-1]) for w in self.vocabulary]
        ]

        predicted_words = difflib.get_close_matches(
            word, filtered_vocab, n=max(2 * n, 100), cutoff=0
        )

        candidates = pd.DataFrame(
            [(w, self.__word_misspelling_score(word, w)) for w in predicted_words],
            columns=["word", "score"],
        ).sort_values("score", ascending=False)

        return candidates[candidates["score"] >= cutoff].head(n).values

    def most_similar_n_words(self, word: str, n: int = 3, cutoff: float = 0.6) -> np.ndarray:
        """Find the n most similar words to the one passed as a parameter."""

        word = word.lower()

        filtered_vocab = self.vocabulary[
            [(word[:2] == w[:2]) for w in self.vocabulary]
        ]

        predicted_words = difflib.get_close_matches(
            word, filtered_vocab, n=max(2 * n, 100), cutoff=0
        )

        candidates = pd.DataFrame(
            [(w, self.__word_similarity_score(word, w)) for w in predicted_words],
            columns=["word", "score"],
        ).sort_values("score", ascending=False)

        return candidates[candidates["score"] >= cutoff].head(n).values

    def __set_language_frequency(self) -> None:
        """Define the frequency dictionary of the chosen language."""
        return get_frequency_dict(lang=self.lang)

    def __set_vocabulary(self) -> None:
        """Define the vocabulary of the chosen language."""

        spacy_vocab = pd.Series(np.array(self.nlp.vocab.strings))

        vocab = pd.Series(np.array(list(self.frequency.keys())))

        return vocab[vocab.isin(spacy_vocab)].values

    def __rule_to_word_selection(self, similar_word_list) -> str:
        """Select the most similar word according to a rule."""
        return similar_word_list[0][0]

    def __get_spacy_model(self, lang: str) -> str:
        """Return the spacy model for the chosen language."""
        if lang == "pt":
            return "pt_core_news_md"

        return lang

    def __get_word_scores(self, word1: str, word2: str) -> Dict[str, Any]:

        direct_similarity = difflib.SequenceMatcher(None, word1, word2).ratio()

        accent_similarity = difflib.SequenceMatcher(
            None,
            unidecode(word1),
            unidecode(word2),
        ).ratio()

        length_similarity = 1 - np.abs(len(word1) - len(word2)) / hmean([len(word1), len(word2)])

        first_letter_similarity = difflib.SequenceMatcher(None, word1[:1], word2[:1]).ratio()

        return {
            "direct_similarity": direct_similarity,
            "accent_similarity": accent_similarity,
            "length_similarity": length_similarity,
            "first_letter_similarity": first_letter_similarity,
        }

    def __word_similarity_score(self, word1: str, word2: str) -> float:
        """Calculate the similarity between two words."""

        scores = self.__get_word_scores(word1, word2)

        return (
            0.5 * scores["direct_similarity"]
            + 0.3 * scores["accent_similarity"]
            + 0.1 * scores["length_similarity"]
            + 0.1 * scores["first_letter_similarity"]
        )

    def __word_misspelling_score(self, word1: str, word2: str) -> float:
        """Calculate the similarity between two words."""

        scores = self.__get_word_scores(word1, word2)

        return (
            0.2 * scores["direct_similarity"]
            + 0.4 * scores["accent_similarity"]
            + 0.2 * scores["length_similarity"]
            + 0.2 * scores["first_letter_similarity"]
        )


def download_spacy_model(model: str) -> spacy.language.Language:
    """
    This function attempts to load a spaCy language model
    specified by the input string "model". If the model
    is not found on the system, the function will download
    it and then return the loaded model.

    Returns:
    spacy.language.Language - The loaded spaCy language model.
    """

    try:
        return spacy.load(model)
    except OSError:
        print(
            "Downloading language model for the spaCy POS tagger\n"
            "(don't worry, this will only happen once)",
            file=stderr,
        )
        from spacy.cli import download

        download(model)
        return spacy.load(model)
