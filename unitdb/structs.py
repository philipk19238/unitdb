from typing import NamedTuple


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


class SemanticSearchResult(NamedTuple):
    """
    NamedTuple to hold the result of a semantic search.
    """

    ref_id: int  # The reference ID of the document.
    distance: int  # The semantic distance from the query.
