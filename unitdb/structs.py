from enum import Enum
from typing import NamedTuple


class DistanceMetric(str, Enum):
    """
    Enum class for defining the distance metrics used in similarity search.
    """

    COSINE = "cosine"  # Cosine similarity metric
    DOT_PRODUCT = "dot_product"  # Dot product similarity metric
    EUCLIDEAN = "euclidean"  # Euclidean distance metric


class SimilaritySearchConfig(NamedTuple):
    """
    NamedTuple for holding the configuration of a similarity search.
    It includes the distance metric to be used in the search.
    """

    distance_metric: DistanceMetric  # The distance metric for the similarity search


class BM25Config(NamedTuple):
    """
    NamedTuple to hold the configuration parameters for the BM25 algorithm.
    """

    k1: float  # Controls how quickly an increase in term frequency leads to term saturation.
    b: float  # Determines the scaling by document length: 0.0 means no scaling, 1.0 means fully scaling.
    delta: float  # Controls the scaling of the BM25+ variant.


class FullTextSearchResult(NamedTuple):
    """
    NamedTuple to hold the result of a full text search.
    """

    ref_id: int  # The reference ID of the document.
    bm25: float  # The BM25 score of the document.


class SimilaritySearchResult(NamedTuple):
    """
    NamedTuple to hold the result of a semantic search.
    """

    ref_id: int  # The reference ID of the document.
    distance: int  # The semantic distance from the query.
