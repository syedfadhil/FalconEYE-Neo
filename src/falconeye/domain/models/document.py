"""Document models for documentation embedding."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import uuid


@dataclass
class DocumentMetadata:
    """
    Metadata for a documentation file.

    Captures information about documentation structure and content.
    """
    file_path: str
    document_type: str  # readme, api_doc, architecture, security_policy, etc.
    title: Optional[str] = None
    sections: List[str] = field(default_factory=list)  # Section headings
    keywords: List[str] = field(default_factory=list)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "document_type": self.document_type,
            "title": self.title,
            "sections": self.sections,
            "keywords": self.keywords,
        }


@dataclass
class DocumentChunk:
    """
    A chunk of documentation content with embedding.

    Similar to CodeChunk but for documentation files.
    """
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        content: str,
        metadata: DocumentMetadata,
        start_char: int,
        end_char: int,
        chunk_index: int,
        total_chunks: int,
    ) -> "DocumentChunk":
        """Create a new document chunk."""
        return cls(
            chunk_id=str(uuid.uuid4()),
            content=content,
            metadata=metadata,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

    def with_embedding(self, embedding: List[float]) -> "DocumentChunk":
        """Return a new chunk with embedding added."""
        return DocumentChunk(
            chunk_id=self.chunk_id,
            content=self.content,
            metadata=self.metadata,
            start_char=self.start_char,
            end_char=self.end_char,
            chunk_index=self.chunk_index,
            total_chunks=self.total_chunks,
            embedding=embedding,
            created_at=self.created_at,
        )

    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Document:
    """
    Represents a documentation file.

    Can be README, API docs, architecture docs, security policies, etc.
    """
    path: Path
    relative_path: str
    content: str
    document_type: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        path: Path,
        relative_path: str,
        content: str,
        document_type: str,
    ) -> "Document":
        """Create a new document."""
        # Extract metadata
        metadata = cls._extract_metadata(relative_path, content, document_type)

        return cls(
            path=path,
            relative_path=relative_path,
            content=content,
            document_type=document_type,
            metadata=metadata,
        )

    @staticmethod
    def _extract_metadata(
        file_path: str,
        content: str,
        document_type: str,
    ) -> DocumentMetadata:
        """Extract metadata from document content."""
        # Extract title (first heading)
        title = None
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("#"):
                title = line.strip().lstrip("#").strip()
                break

        # Extract section headings
        sections = []
        for line in lines:
            if line.strip().startswith("#"):
                heading = line.strip().lstrip("#").strip()
                sections.append(heading)

        # Extract keywords (simple approach - can be enhanced)
        keywords = []
        keyword_indicators = [
            "security", "authentication", "authorization", "api",
            "architecture", "design", "implementation", "configuration"
        ]
        content_lower = content.lower()
        for keyword in keyword_indicators:
            if keyword in content_lower:
                keywords.append(keyword)

        return DocumentMetadata(
            file_path=file_path,
            document_type=document_type,
            title=title,
            sections=sections,
            keywords=keywords,
        )

    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to this document."""
        self.chunks.append(chunk)

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)