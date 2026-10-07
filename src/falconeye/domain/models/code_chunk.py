"""Code chunk models for embedding and analysis."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata associated with a code chunk."""
    file_path: str
    language: str
    start_line: int
    end_line: int
    chunk_index: int
    total_chunks: int
    has_functions: bool = False
    has_imports: bool = False
    function_names: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "has_functions": self.has_functions,
            "has_imports": self.has_imports,
            "function_names": self.function_names,
        }


@dataclass(frozen=True)
class CodeChunk:
    """
    Value object representing a chunk of code for embedding.

    Chunks are created intelligently based on AST boundaries,
    not arbitrary line counts.
    """
    id: UUID
    content: str
    metadata: ChunkMetadata
    token_count: int
    embedding: Optional[list[float]] = None

    @classmethod
    def create(
        cls,
        content: str,
        metadata: ChunkMetadata,
        token_count: int,
        embedding: Optional[list[float]] = None,
    ) -> "CodeChunk":
        """Factory method to create a code chunk."""
        return cls(
            id=uuid4(),
            content=content,
            metadata=metadata,
            token_count=token_count,
            embedding=embedding,
        )

    def with_embedding(self, embedding: list[float]) -> "CodeChunk":
        """Create a new chunk with embedding."""
        return CodeChunk(
            id=self.id,
            content=self.content,
            metadata=self.metadata,
            token_count=self.token_count,
            embedding=embedding,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "token_count": self.token_count,
            "has_embedding": self.embedding is not None,
        }