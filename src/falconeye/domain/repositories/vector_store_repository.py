"""Vector store repository interface (Port)."""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.code_chunk import CodeChunk
from ..models.document import DocumentChunk


class VectorStoreRepository(ABC):
    """
    Port for vector storage operations.

    Implementations provide different backends (ChromaDB, PostgreSQL).
    This interface has NO knowledge of implementation details.
    """

    @abstractmethod
    async def store_chunks(
        self,
        chunks: List[CodeChunk],
        collection: str = "code"
    ) -> None:
        """
        Store code chunks with embeddings.

        Args:
            chunks: List of code chunks with embeddings
            collection: Collection name (code, docs, etc.)
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "code",
        filters: Optional[dict] = None,
    ) -> List[CodeChunk]:
        """
        Search for similar code chunks using semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            collection: Collection to search
            filters: Optional metadata filters

        Returns:
            List of similar code chunks
        """
        pass

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        collection: str = "code",
    ) -> List[CodeChunk]:
        """
        Search using a pre-computed embedding.

        Args:
            embedding: Query embedding vector
            top_k: Number of results
            collection: Collection to search

        Returns:
            List of similar code chunks
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """
        Delete an entire collection.

        Args:
            collection: Collection name to delete
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """
        Check if collection exists.

        Args:
            collection: Collection name

        Returns:
            True if collection exists
        """
        pass

    @abstractmethod
    async def get_chunk_count(self, collection: str) -> int:
        """
        Get number of chunks in collection.

        Args:
            collection: Collection name

        Returns:
            Number of chunks
        """
        pass

    @abstractmethod
    async def store_document_chunks(
        self,
        chunks: List[DocumentChunk],
        collection: str = "documents"
    ) -> None:
        """
        Store document chunks with embeddings.

        Args:
            chunks: List of document chunks with embeddings
            collection: Collection name (defaults to "documents")
        """
        pass

    @abstractmethod
    async def search_similar_documents(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "documents",
        query_embedding: Optional[List[float]] = None,
    ) -> List[DocumentChunk]:
        """
        Search for similar document chunks using semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            collection: Collection to search
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            List of similar document chunks
        """
        pass