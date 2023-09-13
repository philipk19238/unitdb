import abc
from typing import Generic, Iterator, List, Optional, TypeVar

DocumentT = TypeVar("DocumentT", bound="Document")
CorpusT = TypeVar("CorpusT", bound="Corpus")
SearchResultT = TypeVar("SearchResultT")
QueryT = TypeVar("QueryT")


class Document(abc.ABC):
    """
    Base class for a ocument in a corpus.
    """

    def __init__(self, ref_id: int) -> None:
        self._ref_id = ref_id

    @property
    def ref_id(self) -> int:
        """
        Get the reference ID of the document.

        Returns:
            int: The reference ID of the document.
        """
        return self._ref_id


class Corpus(abc.ABC, Generic[DocumentT]):
    """
    Base class for a corpus of documents.
    """

    def __init__(self, documents: List[DocumentT]) -> None:
        self._documents = documents

    def __getitem__(self, ref_id: int) -> DocumentT:
        if ref_id > len(self):
            raise ValueError(f"Reference ID {ref_id} is not in corpus.")
        return self._documents[ref_id]

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[DocumentT]:
        return iter(self._documents)

    @property
    def documents(self) -> List[DocumentT]:
        """
        Get the documents in the corpus.

        Returns:
            List[TextDocument]: The documents in the corpus.
        """
        return self._documents


class Search(abc.ABC, Generic[CorpusT, QueryT, SearchResultT]):
    """
    Base class for search.
    """

    def __init__(self, corpus: CorpusT) -> None:
        self._corpus = corpus

    @abc.abstractmethod
    def search(self, query: QueryT, ref_ids: Optional[List[int]] = None) -> List[SearchResultT]:
        """
        This is an abstract method that should be implemented in subclasses.
        It is intended to search the corpus with a given query and return a list of search results.

        Args:
            query (QueryT): The query to search the corpus with.
            ref_ids (Optional[List[int]]): A list of IDs to limit the search to. If None, the entire corpus is searched.

        Returns:
            List[SearchResultT]: The search results. Thesee results depends on the implementation in the subclass.
        """
        ...
