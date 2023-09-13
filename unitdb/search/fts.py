import abc
import math
from collections import defaultdict
from functools import cached_property
from typing import Dict, Generic, Iterator, List, Optional, Set, TypeVar

import numpy as np

from unitdb.search.base import Corpus, Document, Search
from unitdb.structs import BM25Config, FullTextSearchResult

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class TextDocument(Document):
    """
    Class to represent a text document.
    Each document is represented as a list of string tokens.
    """

    def __init__(self, ref_id: int, tokens: List[str]) -> None:
        """
        Initialize a TextDocument instance.

        Args:
            ref_id (int): The reference ID of the document.
            tokens (List[str]): The list of string tokens representing the document.
        """
        super().__init__(ref_id)
        self._tokens: List[str] = tokens

    def __len__(self) -> int:
        """
        Get the length of the document (number of tokens).

        Returns:
            int: The number of tokens in the document.
        """
        return len(self._tokens)

    def __iter__(self) -> Iterator[str]:
        """
        Get an iterator over the tokens in the document.

        Returns:
            Iterator[str]: An iterator over the tokens in the document.
        """
        return iter(self._tokens)

    @property
    def tokens(self) -> List[str]:
        """
        Get the tokens of the document.

        Returns:
            List[str]: The tokens of the document.
        """
        return self._tokens

    @cached_property
    def unique_tokens(self) -> Set[str]:
        """
        Get the unique tokens in the document.

        Returns:
            Set[str]: The set of unique tokens in the document.
        """
        return set(self._tokens)


class TextCorpus(Corpus[TextDocument]):
    """
    Class to represent a corpus of documents.
    Each corpus is represented as a list of TextDocument instances.
    """

    @cached_property
    def doc_lengths(self) -> np.ndarray:
        """
        Get the lengths of the documents in the corpus.

        Returns:
            np.ndarray: An array of the lengths of the documents in the corpus.
        """
        return np.array([len(doc) for doc in self._documents])

    @cached_property
    def avg_dl(self) -> float:
        """
        Get the average document length in the corpus.

        Returns:
            float: The average document length in the corpus.
        """
        return np.sum(self.doc_lengths) / len(self._documents)


class Index(abc.ABC, Generic[KeyT, ValueT]):
    """
    Abstract base class for creating an index from a corpus.
    Each index is represented as a dictionary mapping keys to values.
    """

    def __init__(self, corpus: TextCorpus) -> None:
        """
        Initialize an Index instance.

        Args:
            corpus (Corpus): The corpus to index.
        """
        self._corpus: TextCorpus = corpus
        self._index: Dict[KeyT, ValueT] = self.build_index()

    @abc.abstractmethod
    def build_index(self) -> Dict[KeyT, ValueT]:
        """
        Abstract method to build the index.

        Returns:
            Dict[KeyT, ValueT]: The built index.
        """
        ...

    @property
    def index(self) -> Dict[KeyT, ValueT]:
        """
        Get the index.

        Returns:
            Dict[KeyT, ValueT]: The index.
        """
        return self._index


class IDFIndex(Index[str, float]):
    """
    Class for creating an index of inverse document frequencies from a list of documents.
    Each index is represented as a dictionary mapping tokens to their inverse document frequencies.
    """

    def build_index(self) -> Dict[str, float]:
        """
        Build the index of inverse document frequencies.

        Returns:
            Dict[str, float]: The built index.
        """
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
        """
        Get the inverse document frequency of a token.

        Args:
            token (str): The token to get the inverse document frequency of.

        Returns:
            float: The inverse document frequency of the token.
        """
        return self._index.get(token, 0)


class InvertedIndex(Index[int, Dict[str, int]]):
    """
    Class for creating an index of token counts from a list of documents.
    Each index is represented as a dictionary mapping document reference IDs to dictionaries of token counts.
    """

    def build_index(self) -> Dict[int, Dict[str, int]]:
        """
        Build the index of token counts.

        Returns:
            Dict[int, Dict[str, int]]: The built index.
        """
        inv_index: defaultdict[int, Dict[str, int]] = defaultdict(dict)
        for document in self._corpus.documents:
            for token in document.tokens:
                if token not in inv_index[document.ref_id]:
                    inv_index[document.ref_id][token] = 0
                inv_index[document.ref_id][token] += 1
        return inv_index

    def get_token_frequency(self, token: str, ref_id: int) -> int:
        """
        Get the frequency of a token in a document.

        Args:
            token (str): The token to get the frequency of.
            ref_id (int): The reference ID of the document to get the token frequency in.

        Returns:
            int: The frequency of the token in the document.
        """
        return self._index[ref_id].get(token, 0)


class FullTextSearch(Search[TextCorpus, List[str], FullTextSearchResult]):
    """
    Class for performing full text search on a corpus.
    Each search is performed using the BM25 algorithm.
    """

    def __init__(self, corpus: TextCorpus, config: BM25Config) -> None:
        """
        Initialize a FullTextSearch instance.

        Args:
            corpus (Corpus): The corpus to search in.
            config (BM25Config): The configuration parameters for the BM25 algorithm.
        """
        super().__init__(corpus)
        self._config: BM25Config = config
        self._idf: IDFIndex = IDFIndex(corpus)
        self._inv: InvertedIndex = InvertedIndex(corpus)

    def search(self, query: List[str], ref_ids: Optional[List[int]] = None) -> List[FullTextSearchResult]:
        """
        Performs a full-text search on the corpus using the BM25 algorithm.

        The BM25 algorithm calculates the relevance of each document in the corpus to the query.
        The results are sorted by relevance in descending order.

        Args:
            query (List[str]): The list of tokens to search for in the corpus.
            ref_ids (Optional[List[int]]): The list of reference IDs of the documents to search in.
                                            If None, all documents in the corpus are searched.

        Returns:
            List[FullTextSearchResult]: A list of FullTextSearchResult objects, each representing a document
                                        and its relevance to the query. The list is sorted by relevance in
                                        descending order.

        Raises:
            ValueError: If any reference ID in ref_ids is not less than the length of the corpus.
        """
        if ref_ids is None:
            ref_ids = list(range(len(self._corpus)))
            doc_lengths = self._corpus.doc_lengths
        else:
            try:
                assert all(ref_id < len(self._corpus) for ref_id in ref_ids)
            except AssertionError:
                raise ValueError("All reference IDs must be less than the length of the corpus.")
            doc_lengths = self._corpus.doc_lengths[ref_ids]

        score: np.ndarray = np.zeros(len(ref_ids))
        q_freq: np.ndarray = np.array([self._inv.get_token_frequency(q, ref_id) for q in query for ref_id in ref_ids])
        ctd: np.ndarray = q_freq / (1 - self._config.b + self._config.b * doc_lengths / self._corpus.avg_dl)
        score += (
            np.array([self._idf.get_idf(q) for q in query])
            * (self._config.k1 + 1)
            * (ctd + self._config.delta)
            / (self._config.k1 + ctd + self._config.delta)
        )
        results: List[FullTextSearchResult] = [
            FullTextSearchResult(ref_id=ref_id, bm25=doc_score) for ref_id, doc_score in zip(ref_ids, score)
        ]
        return sorted(results, key=lambda x: x.bm25, reverse=True)
