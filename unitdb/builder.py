import abc
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

import polars as pl
from polars.expr.expr import Expr

from unitdb.constants import DEFAULT_ALPHA
from unitdb.embedding import call_openai
from unitdb.structs import FullTextSearchResult, SimilaritySearchResult
from unitdb.tokenizer import Tokenizer

if TYPE_CHECKING:
    from unitdb.db import Collection


class HybridRank(NamedTuple):
    """
    A NamedTuple representing the hybrid rank of a document.
    """

    ref_id: int
    weighted_score: float


class ExecutorMixin:
    """
    A mixin class that provides methods for executing keyword and semantic searches.
    """

    def _execute_keyword(
        self, collection: "Collection", query: str, ref_ids: Optional[List[int]]
    ) -> List[FullTextSearchResult]:
        """
        Execute a keyword search on the given collection.
        """
        tokenized: List[str] = Tokenizer.tokenize(query)
        return collection.fts_engine.search(tokenized, ref_ids)

    def _execute_semantic(
        self, collection: "Collection", query: str, ref_ids: Optional[List[int]]
    ) -> List[SimilaritySearchResult]:
        """
        Execute a semantic search on the given collection.
        """
        embedding: List[float] = call_openai([query])[0]
        return collection.sim_engine.search(embedding, ref_ids)


class Query(abc.ABC, ExecutorMixin):
    """
    An abstract base class for different types of queries.
    """

    def __init__(self, collection: "Collection"):
        """
        Initialize a Query instance.
        """
        self._collection: "Collection" = collection
        self._filters: List[Expr] = []
        self._limit: int = 100
        self._query: Optional[str] = None

    def with_filter(self, expr: Expr) -> "Query":
        """
        Add a filter to the query.
        """
        self._filters.append(expr)
        return self

    def with_limit(self, limit: int) -> "Query":
        """
        Set the limit for the number of results to return.
        """
        self._limit = limit
        return self

    def execute(self) -> List[Dict[str, Any]]:
        """
        Execute the query and return the results.
        """
        if not self._query:
            raise ValueError("Query is empty!")
        ref_ids: Optional[List[int]] = None
        if self._filters:
            ref_ids = self._apply_filters()
        result_ref_ids: List[int] = self._execute_query(self._query, ref_ids)
        result_ref_ids = result_ref_ids[: min(self._limit, len(result_ref_ids))]
        order_index_table: Dict[int, int] = {ref_id: index for index, ref_id in enumerate(result_ref_ids)}
        result_df: pl.DataFrame = self._collection.df.filter(pl.col("ref_id").is_in(result_ref_ids))
        result_dict: List[Dict[str, Any]] = result_df.to_dicts()
        result_dict.sort(key=lambda x: order_index_table[x["ref_id"]])
        return result_dict

    def _apply_filters(self) -> List[int]:
        """
        Apply the filters to the query.
        """
        combined_filters: Any = reduce(operator.and_, self._filters)
        filtered_df: pl.DataFrame = self._collection.df.filter(combined_filters)
        return filtered_df["ref_id"].to_list()

    @abc.abstractmethod
    def _execute_query(self, query: str, ref_ids: Optional[List[int]]) -> List[int]:
        """
        Abstract method to execute the query.
        """
        ...


class HybridQuery(Query):
    """
    A class for executing hybrid queries that combine keyword and semantic searches.
    """

    def __init__(self, collection: "Collection"):
        """
        Initialize a HybridQuery instance.
        """
        super().__init__(collection)
        self._alpha: float = DEFAULT_ALPHA

    def hybrid(self, query: str, alpha: float) -> "HybridQuery":
        """
        Set the query and alpha value for the hybrid search.
        """
        self._query = query
        self._alpha = alpha
        return self

    def _weighted_rerank(
        self, fts_results: List[FullTextSearchResult], sim_results: List[SimilaritySearchResult]
    ) -> List[int]:
        """
        Rerank the results based on a weighted combination of keyword and semantic search ranks.
        """
        fts_ref_map = {result.ref_id: index for index, result in enumerate(fts_results)}
        sim_ref_map = {result.ref_id: index for index, result in enumerate(sim_results)}
        weighted_scores: List[HybridRank] = []
        for ref_id in fts_ref_map:
            fts_rank: int = fts_ref_map[ref_id]
            sim_rank: int = sim_ref_map[ref_id]
            score: float = (self._alpha * sim_rank) + ((1 - self._alpha) * fts_rank)
            weighted_scores.append(HybridRank(ref_id, score))
        weighted_scores.sort(key=lambda x: x.weighted_score)
        return [score.ref_id for score in weighted_scores]

    def _execute_query(self, query: str, ref_ids: Optional[List[int]]) -> List[int]:
        """
        Execute the hybrid query and return the results.
        """
        fts_results: List[FullTextSearchResult] = self._execute_keyword(self._collection, query, ref_ids)
        sim_results: List[SimilaritySearchResult] = self._execute_semantic(self._collection, query, ref_ids)
        return self._weighted_rerank(fts_results, sim_results)


class KeywordQuery(Query):
    """
    A class for executing keyword queries.
    """

    def keyword(self, query: str) -> "KeywordQuery":
        """
        Set the query for the keyword search.
        """
        self._query = query
        return self

    def _execute_query(self, query: str, ref_ids: Optional[List[int]]) -> List[int]:
        """
        Execute the keyword query and return the results.
        """
        keyword_results: List[FullTextSearchResult] = self._execute_keyword(self._collection, query, ref_ids)
        return [result.ref_id for result in keyword_results]


class SemanticQuery(Query):
    """
    A class for executing semantic queries.
    """

    def semantic(self, query: str) -> "SemanticQuery":
        """
        Set the query for the semantic search.
        """
        self._query = query
        return self

    def _execute_query(self, query: str, ref_ids: Optional[List[int]]) -> List[int]:
        """
        Execute the semantic query and return the results.
        """
        semantic_results: List[SimilaritySearchResult] = self._execute_semantic(self._collection, query, ref_ids)
        return [result.ref_id for result in semantic_results]
