from functools import cached_property
from typing import List, Optional

import numpy as np

from unitdb.search.base import Corpus, Document, Search
from unitdb.structs import (
    DistanceMetric,
    SimilaritySearchConfig,
    SimilaritySearchResult,
)


class EmbeddingDocument(Document):
    """
    Class to represent an embedding document.
    Each document is represented as a list of floats.
    """

    def __init__(self, ref_id: int, vector: List[float]) -> None:
        super().__init__(ref_id)
        self._vector: List[float] = vector

    @property
    def vector(self) -> List[float]:
        return self._vector


class EmbeddingCorpus(Corpus[EmbeddingDocument]):
    """
    Class to represent a corpus of embedding documents.
    Each corpus is represented as a list of EmbeddingDocument instances.
    """

    @cached_property
    def vectors(self) -> np.ndarray:
        return np.array([doc.vector for doc in self.documents])


class SimilaritySearch(Search[EmbeddingCorpus, List[float], SimilaritySearchResult]):
    def __init__(self, corpus: EmbeddingCorpus, config: SimilaritySearchConfig) -> None:
        super().__init__(corpus)
        self._config: SimilaritySearchConfig = config

    def search(self, query: List[float], ref_ids: Optional[List[int]] = None) -> List[SimilaritySearchResult]:
        """
        Conducts a similarity search on the corpus using the provided query and reference IDs.

        Args:
            query (List[float]): The query vector.
            ref_ids (Optional[List[int]]): The reference IDs for the documents in the corpus.
                If None, all documents in the corpus are considered.

        Returns:
            List[SimilaritySearchResult]: A list of search results, each represented as a SimilaritySearchResult
                instance. The results are sorted in descending order of similarity.
        """
        if ref_ids is None:
            ref_ids = list(range(len(self._corpus)))
        query_vector: np.ndarray = np.array(query)
        corpus_vectors: np.ndarray = self._corpus.vectors[ref_ids]

        distances: np.ndarray
        if self._config.distance_metric == DistanceMetric.COSINE:
            distances = -np.linalg.norm(corpus_vectors - query_vector, axis=1)
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            distances = np.dot(corpus_vectors, query_vector)
        elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
            norms = np.linalg.norm(corpus_vectors, axis=1) * np.linalg.norm(query_vector)
            distances = np.dot(corpus_vectors, query_vector) / norms
        else:
            raise ValueError("Unsupported distance metric: {}".format(self._config.distance_metric))

        results = [SimilaritySearchResult(ref_id, distance) for ref_id, distance in zip(ref_ids, distances)]

        results.sort(key=lambda result: result.distance, reverse=True)
        return results
