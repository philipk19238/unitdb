import os
import tempfile
from typing import Any, Dict, List, Optional

import polars as pl

from unitdb.builder import HybridQuery, KeywordQuery, SemanticQuery
from unitdb.constants import (
    DEFAULT_ALPHA,
    DEFAULT_B,
    DEFAULT_CSV_PATH,
    DEFAULT_DELTA,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_K1,
    VECTOR_COLUMN_NAME,
)
from unitdb.search.fts import FullTextSearch, TextCorpus
from unitdb.search.sim import EmbeddingCorpus, SimilaritySearch
from unitdb.structs import BM25Config, DistanceMetric, SimilaritySearchConfig


class Collection:
    """
    A class used to represent a collection of documents for search.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        vector_column_name: str = VECTOR_COLUMN_NAME,
        columns_to_index: Optional[List[str]] = None,
        bm25_config: BM25Config = BM25Config(k1=DEFAULT_K1, b=DEFAULT_B, delta=DEFAULT_DELTA),
        similarity_search_config: SimilaritySearchConfig = SimilaritySearchConfig(
            distance_metric=DistanceMetric[DEFAULT_DISTANCE_METRIC]
        ),
    ):
        """
        Initializes the Collection with the given parameters.
        """
        self._df = df
        self._vector_column_name: str = vector_column_name
        self._columns_to_index: Optional[List[str]] = columns_to_index
        self._bm25_config: BM25Config = bm25_config
        self._similarity_search_config: SimilaritySearchConfig = similarity_search_config
        self._fts = self.initialize_fts(self._df, columns_to_index, bm25_config)
        self._sim = self.initialize_sim(self._df, vector_column_name, similarity_search_config)

    @property
    def name(self) -> str:
        """
        Returns the name of the Collection.
        """
        return self.name

    @property
    def fts_engine(self) -> FullTextSearch:
        """
        Returns the FullTextSearch engine of the Collection.
        """
        return self._fts

    @property
    def sim_engine(self) -> SimilaritySearch:
        """
        Returns the SimilaritySearch engine of the Collection.
        """
        return self._sim

    @property
    def df(self) -> pl.DataFrame:
        """
        Returns the DataFrame of the Collection.
        """
        return self._df

    def initialize_fts(
        self, df: pl.DataFrame, columns_to_index: Optional[List[str]], bm25_config: BM25Config
    ) -> FullTextSearch:
        """
        Initializes the FullTextSearch engine with the given parameters.
        """
        corpus: TextCorpus = TextCorpus.from_dataframe(df, columns_to_index)
        return FullTextSearch(corpus, bm25_config)

    def initialize_sim(
        self, df: pl.DataFrame, vector_column_name: str, similarity_search_config: SimilaritySearchConfig
    ) -> SimilaritySearch:
        """
        Initializes the SimilaritySearch engine with the given parameters.
        """
        corpus: EmbeddingCorpus = EmbeddingCorpus.from_dataframe(df, vector_column_name)
        return SimilaritySearch(corpus, similarity_search_config)

    def keyword(self, query: str) -> KeywordQuery:
        """
        Returns a KeywordQuery object with the given query.
        """
        return KeywordQuery(self).keyword(query)

    def semantic(self, query: str) -> SemanticQuery:
        """
        Returns a SemanticQuery object with the given query.
        """
        return SemanticQuery(self).semantic(query)

    def hybrid(self, query: str, alpha: float = DEFAULT_ALPHA) -> HybridQuery:
        """
        Returns a HybridQuery object with the given query and alpha.
        """
        return HybridQuery(self).hybrid(query, alpha)

    @classmethod
    def verify_and_index_df(cls, df: pl.DataFrame, vector_column_name: str) -> pl.DataFrame:
        """
        Verifies the DataFrame and indexes it.
        """
        if vector_column_name not in df.columns:
            raise ValueError(f"Vector column {vector_column_name} not found in index!")
        return df.with_row_count("ref_id")

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        vector_column_name: str = VECTOR_COLUMN_NAME,
        columns_to_index: Optional[List[str]] = None,
        bm25_config: BM25Config = BM25Config(k1=DEFAULT_K1, b=DEFAULT_B, delta=DEFAULT_DELTA),
        similarity_search_config: SimilaritySearchConfig = SimilaritySearchConfig(
            distance_metric=DistanceMetric[DEFAULT_DISTANCE_METRIC]
        ),
    ) -> "Collection":
        """
        Creates a Collection object from a CSV file.
        """
        df: pl.DataFrame = pl.read_csv(csv_path)
        return Collection(
            df=cls.verify_and_index_df(df, vector_column_name),
            vector_column_name=vector_column_name,
            columns_to_index=columns_to_index,
            bm25_config=bm25_config,
            similarity_search_config=similarity_search_config,
        )

    @classmethod
    def from_dicts(
        cls,
        dicts: List[Dict[str, Any]],
        vector_column_name: str = VECTOR_COLUMN_NAME,
        columns_to_index: Optional[List[str]] = None,
        bm25_config: BM25Config = BM25Config(k1=DEFAULT_K1, b=DEFAULT_B, delta=DEFAULT_DELTA),
        similarity_search_config: SimilaritySearchConfig = SimilaritySearchConfig(
            distance_metric=DistanceMetric[DEFAULT_DISTANCE_METRIC]
        ),
    ) -> "Collection":
        """
        Creates a Collection object from a list of dictionaries.
        """
        df: pl.DataFrame = pl.DataFrame(dicts)
        return Collection(
            df=cls.verify_and_index_df(df, vector_column_name),
            vector_column_name=vector_column_name,
            columns_to_index=columns_to_index,
            bm25_config=bm25_config,
            similarity_search_config=similarity_search_config,
        )

    @classmethod
    def from_s3(
        self,
        bucket: str,
        key: str,
        vector_column_name: str = VECTOR_COLUMN_NAME,
        columns_to_index: Optional[List[str]] = None,
        bm25_config: BM25Config = BM25Config(k1=DEFAULT_K1, b=DEFAULT_B, delta=DEFAULT_DELTA),
        similarity_search_config: SimilaritySearchConfig = SimilaritySearchConfig(
            distance_metric=DistanceMetric[DEFAULT_DISTANCE_METRIC]
        ),
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY"),
    ) -> "Collection":
        """
        Creates a Collection object from a file in an S3 bucket.
        """
        try:
            import boto3
            from boto3.s3.transfer import S3Transfer, TransferConfig
        except ImportError:
            raise ImportError("Boto3 is required for S3 support. Install using pip install boto3.")
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        config = TransferConfig(
            multipart_threshold=1024 * 1024 * 10,
            max_concurrency=10,
            multipart_chunksize=1024 * 5,
            use_threads=True,
        )
        transfer = S3Transfer(client=client, config=config)
        with tempfile.TemporaryDirectory() as tmpdir:
            transfer.download_file(
                bucket,
                key,
                os.path.join(tmpdir, os.path.basename(DEFAULT_CSV_PATH)),
            )
            return Collection.from_csv(
                os.path.join(tmpdir, os.path.basename(DEFAULT_CSV_PATH)),
                vector_column_name=vector_column_name,
                columns_to_index=columns_to_index,
                bm25_config=bm25_config,
                similarity_search_config=similarity_search_config,
            )
