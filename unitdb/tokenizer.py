import re
import string
from re import Pattern
from typing import List, Set

from nltk.stem.porter import PorterStemmer


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
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }

    ALPHANUM: Pattern[str] = re.compile(r"^\d*[a-z][\-.0-9:_a-z]{1,}$")

    STEMMER: PorterStemmer = PorterStemmer()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = [token.strip(string.punctuation) for token in text.split()]
        tokens = [x for x in tokens if x not in cls.STOP_WORDS]
        tokens = [token for token in tokens if re.match(cls.ALPHANUM, token)]
        tokens = [cls.STEMMER.stem(x) for x in tokens]
        return tokens
