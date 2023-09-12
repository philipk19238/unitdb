import abc
import math
from collections import defaultdict
from functools import cached_property
from typing import Dict, Generic, Iterator, List, Set, TypeVar

import numpy as np

from unitdb.structs import BM25Config, FullTextSearchResult

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class TextDocument:
    """Wrapper around a list of string tokens."""

    def __init__(self, ref_id: int, tokens: List[str]) -> None:
        self._ref_id = ref_id
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    @property
    def ref_id(self) -> int:
        return self._ref_id

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @cached_property
    def unique_tokens(self) -> Set[str]:
        return set(self._tokens)


class Corpus:
    """Class that wraps a list of documents."""

    def __init__(self, documents: List[TextDocument]) -> None:
        self._documents: List[TextDocument] = documents

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[TextDocument]:
        return iter(self._documents)

    @property
    def documents(self) -> List[TextDocument]:
        return self._documents

    @cached_property
    def doc_lengths(self) -> np.ndarray:
        return np.array([len(doc) for doc in self._documents])

    @cached_property
    def avg_dl(self) -> float:
        return np.sum(self.doc_lengths) / len(self._documents)


class Index(abc.ABC, Generic[KeyT, ValueT]):
    def __init__(self, corpus: Corpus) -> None:
        self._corpus: Corpus = corpus
        self._index: Dict[KeyT, ValueT] = self.build_index()

    @abc.abstractmethod
    def build_index(self) -> Dict[KeyT, ValueT]:
        pass

    @property
    def index(self) -> Dict[KeyT, ValueT]:
        return self._index


class IDFIndex(Index[str, float]):
    """Class for creating an index of inverse document frequencies from a list of documents."""

    def build_index(self) -> Dict[str, float]:
        f_map: defaultdict[str, int] = defaultdict(int)
        for document in self._corpus.documents:
            for token in document.unique_tokens:
                f_map[token] += 1
        idf_index: defaultdict[str, float] = defaultdict(float)
        for token, frequency in f_map.items():
            idf: float = math.log(len(self._corpus) + 1) - math.log(frequency + 0.5)
            idf_index[token] = idf
        return idf_index

    def get_idf(self, token: str) -> float:
        return self._index.get(token, 0)


class InvertedIndex(Index[int, Dict[str, int]]):
    """Class for creating an index of token counts from a list of documents."""

    def build_index(self) -> Dict[int, Dict[str, int]]:
        inv_index: defaultdict[int, Dict[str, int]] = defaultdict(dict)
        for document in self._corpus.documents:
            for token in document.tokens:
                if token not in inv_index[document.ref_id]:
                    inv_index[document.ref_id][token] = 0
                inv_index[document.ref_id][token] += 1
        return inv_index

    def get_frequency(self, token: str, ref_id: int) -> int:
        return self._index[ref_id].get(token, 0)


class FullTextSearch:
    def __init__(self, corpus: Corpus, config: BM25Config) -> None:
        self._corpus = corpus
        self._config = config
        self._idf = IDFIndex(corpus)
        self._inv = InvertedIndex(corpus)

    def search(self, query: TextDocument) -> List[FullTextSearchResult]:
        score: np.ndarray = np.zeros(len(self._corpus))
        q_freq: np.ndarray = np.array(
            [self._inv.get_frequency(q, doc.ref_id) for q in query.tokens for doc in self._corpus]
        )
        ctd: np.ndarray = q_freq / (
            1 - self._config.b + self._config.b * self._corpus.doc_lengths / self._corpus.avg_dl
        )
        score += (
            (np.array([self._idf.get_idf(q) for q in query.tokens]) or 0)
            * (self._config.k1 + 1)
            * (ctd + self._config.delta)
            / (self._config.k1 + ctd + self._config.delta)
        )
        results: List[FullTextSearchResult] = [
            FullTextSearchResult(ref_id=doc.ref_id, bm25=doc_score) for doc, doc_score in zip(self._corpus, score)
        ]
        return sorted(results, key=lambda x: x.bm25, reverse=True)
