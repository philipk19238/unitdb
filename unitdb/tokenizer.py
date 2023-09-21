import string
from typing import List, Set

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")


class Tokenizer:
    STOP_WORDS: Set[str] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there" "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }

    LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        This method tokenizes the input text into a list of words. It performs the following steps:
        1. Converts the text to lower case.
        2. Splits the text into tokens, removing punctuation.
        3. Removes stop words from the tokens.
        4. Filters out tokens that do not match the ALPHANUM pattern.
        5. Stems the tokens using the PorterStemmer.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: The list of tokenized and processed words.
        """
        text = text.lower()
        tokens: List[str] = [token.strip(string.punctuation) for token in text.split()]
        tokens = [x for x in tokens if x not in cls.STOP_WORDS]
        tokens = [cls.LEMMATIZER.lemmatize(x) for x in tokens]
        return tokens
