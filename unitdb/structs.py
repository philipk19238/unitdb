from typing import NamedTuple


class BM25Config(NamedTuple):
    k1: float
    b: float
    delta: float


class FullTextSearchResult(NamedTuple):
    ref_id: int
    bm25: float


class SemanticSearchResult(NamedTuple):
    ref_id: int
    distance: int
