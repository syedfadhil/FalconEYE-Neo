"""Context assembler domain service."""

import asyncio
from typing import List, Optional, Dict, Any
import time
from ..models.prompt import PromptContext
from ..repositories.vector_store_repository import VectorStoreRepository
from ..repositories.metadata_repository import MetadataRepository
from .llm_service import LLMService
from ...infrastructure.logging import FalconEyeLogger


# embeddinggemma:300m and similar small embedding models have a ~2048 token
# context window. Truncate the query before generating an embedding so the
# LLM does not reject over-length input.
_EMBEDDING_QUERY_CHAR_LIMIT = 2048


class ContextAssembler:
    """
    Domain service for assembling rich context for AI analysis.

    Gathers everything the LLM needs to reason about a file:
    - Code to analyze
    - Structural metadata (AST)
    - Related code (RAG)
    - Related documentation (RAG)

    NO pattern matching - just context assembly for the AI.
    """

    def __init__(
        self,
        vector_store: VectorStoreRepository,
        metadata_repo: MetadataRepository,
        llm_service: Optional[LLMService] = None,
    ):
        """
        Initialize context assembler.

        Args:
            vector_store: Vector store for semantic search
            metadata_repo: Metadata repository for structural info
            llm_service: LLM service for embedding generation. Passing this
                in is strongly preferred over the legacy path where the
                assembler created its own ad-hoc adapter, because that
                path (a) constructed a fresh Ollama client + circuit
                breaker per RAG call and (b) generated the same query
                embedding twice per file.
        """
        self.vector_store = vector_store
        self.metadata_repo = metadata_repo
        self.llm_service = llm_service
        self.logger = FalconEyeLogger.get_instance()

    async def assemble_context(
        self,
        file_path: str,
        code_snippet: str,
        language: str,
        top_k_similar: int = 5,
        top_k_docs: int = 3,
        original_file: Optional[str] = None,
        analysis_type: str = "review",
    ) -> PromptContext:
        """
        Assemble comprehensive context for AI analysis.

        Performance notes:
        - The query embedding is generated exactly once and shared between
          the code-similarity search and the documentation-similarity
          search. Previously each helper generated its own embedding for
          the same input.
        - Metadata retrieval and both RAG searches run concurrently via
          asyncio.gather. Previously they were awaited sequentially.
        """
        start_time = time.time()

        self.logger.debug(
            "Starting context assembly",
            extra={
                "file_path": file_path,
                "language": language,
                "code_size": len(code_snippet),
                "top_k_similar": top_k_similar,
                "analysis_type": analysis_type,
            }
        )

        # Compute the shared query embedding once.
        query_embedding = await self._compute_query_embedding(code_snippet)

        # Fan out: metadata + code RAG + docs RAG all in parallel.
        structural_metadata, related_code, related_docs = await asyncio.gather(
            self._get_structural_metadata(file_path),
            self._get_related_code(
                code_snippet,
                file_path,
                top_k_similar,
                query_embedding=query_embedding,
            ),
            self._get_related_documentation(
                code_snippet,
                top_k_docs,
                query_embedding=query_embedding,
            ),
            return_exceptions=False,
        )

        context = PromptContext(
            file_path=file_path,
            code_snippet=code_snippet,
            language=language,
            structural_metadata=structural_metadata,
            related_code=related_code,
            related_docs=related_docs,
            original_file=original_file,
            analysis_type=analysis_type,
        )

        duration = time.time() - start_time
        self.logger.debug(
            "Context assembly completed",
            extra={
                "file_path": file_path,
                "has_metadata": structural_metadata is not None,
                "has_related_code": related_code is not None,
                "has_related_docs": related_docs is not None,
                "duration_seconds": round(duration, 2),
            }
        )

        return context

    async def _compute_query_embedding(
        self, code_snippet: str
    ) -> Optional[List[float]]:
        """
        Generate the shared query embedding used by both RAG searches.

        Falls back to None if no LLM service was injected (legacy path);
        callers will then fall back to ChromaDB's default text query.
        """
        if self.llm_service is None:
            self.logger.warning(
                "ContextAssembler has no llm_service; RAG will use "
                "text-query fallback (slower and less consistent)"
            )
            return None

        truncated = (
            code_snippet[:_EMBEDDING_QUERY_CHAR_LIMIT]
            if len(code_snippet) > _EMBEDDING_QUERY_CHAR_LIMIT
            else code_snippet
        )
        try:
            return await self.llm_service.generate_embedding(truncated)
        except Exception as e:
            # Don't break analysis just because we couldn't build the
            # RAG query. The LLM can still reason about the file itself.
            self.logger.warning(
                "Failed to generate query embedding; RAG will be skipped",
                extra={"error": str(e)},
            )
            return None

    async def _get_structural_metadata(
        self,
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve structural metadata for the file."""
        try:
            metadata = await self.metadata_repo.get_metadata(file_path)
            if metadata:
                return metadata.to_dict()
            return None
        except Exception as e:
            self.logger.warning(
                "Failed to retrieve structural metadata",
                extra={"file_path": file_path, "error": str(e)},
                exc_info=True,
            )
            return None

    async def _get_related_code(
        self,
        code_snippet: str,
        current_file: str,
        top_k: int,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        """
        Use RAG to find related code chunks.

        Accepts a pre-computed query_embedding so the same embedding can be
        shared with the docs search. If not provided, falls back to text
        query (uses ChromaDB's default embedding).
        """
        try:
            similar_chunks = await self.vector_store.search_similar(
                query=code_snippet,
                top_k=top_k + 5,  # Buffer for filtering
                collection="code",
                query_embedding=query_embedding,
            )

            # Filter current file out and deduplicate identical chunks
            filtered_chunks = []
            seen_chunks = set()
            for chunk in similar_chunks:
                if chunk.metadata.file_path == current_file:
                    continue
                dedupe_key = (chunk.metadata.file_path, chunk.content)
                if dedupe_key in seen_chunks:
                    continue
                seen_chunks.add(dedupe_key)
                filtered_chunks.append(chunk)
                if len(filtered_chunks) >= top_k:
                    break

            if not filtered_chunks:
                return None

            related_parts = []
            for i, chunk in enumerate(filtered_chunks, 1):
                related_parts.append(
                    f"[Related Code {i}] From {chunk.metadata.file_path}:\n"
                    f"{chunk.content}\n"
                )
            return "\n".join(related_parts)

        except Exception as e:
            self.logger.warning(
                "Failed to retrieve related code",
                extra={"current_file": current_file, "error": str(e)},
                exc_info=True,
            )
            return None

    async def _get_related_documentation(
        self,
        code_snippet: str,
        top_k: int,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        """
        Use RAG to find relevant documentation.

        Accepts a pre-computed query_embedding shared with the code
        search. Falls back to text query if not provided.
        """
        try:
            doc_chunks = await self.vector_store.search_similar_documents(
                query=code_snippet,
                top_k=top_k,
                collection="documents",
                query_embedding=query_embedding,
            )

            if not doc_chunks:
                return None

            doc_parts = []
            for i, chunk in enumerate(doc_chunks, 1):
                doc_type = chunk.metadata.document_type.replace("_", " ").title()
                doc_parts.append(
                    f"[Documentation {i}] {doc_type} - {chunk.metadata.file_path}:\n"
                    f"{chunk.content}\n"
                )
            return "\n".join(doc_parts)

        except Exception as e:
            self.logger.warning(
                "Failed to retrieve related documentation",
                extra={"error": str(e)},
                exc_info=True,
            )
            return None

    async def assemble_multi_file_context(
        self,
        file_contexts: List[tuple[str, str, str]],  # (path, code, language)
        top_k_per_file: int = 3,
        concurrency: int = 4,
    ) -> List[PromptContext]:
        """
        Assemble contexts for multiple files concurrently.

        Args:
            file_contexts: List of (file_path, code, language) tuples
            top_k_per_file: Similar chunks per file
            concurrency: Max concurrent assembles (bounded to avoid
                overwhelming Ollama's embedding model with parallel
                requests it can't service).

        Returns:
            List of PromptContext objects in input order.
        """
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(item):
            file_path, code, language = item
            async with sem:
                return await self.assemble_context(
                    file_path=file_path,
                    code_snippet=code,
                    language=language,
                    top_k_similar=top_k_per_file,
                )

        return await asyncio.gather(*(_one(item) for item in file_contexts))
