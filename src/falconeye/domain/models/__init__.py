"""Domain models - Entities and Value Objects."""

from .codebase import Codebase, CodeFile
from .security import SecurityFinding, SecurityReview, Severity, FindingConfidence
from .code_chunk import CodeChunk, ChunkMetadata
from .structural import StructuralMetadata, FunctionInfo, ImportInfo, CallInfo, ClassInfo
from .prompt import PromptContext, PromptTemplate

__all__ = [
    "Codebase",
    "CodeFile",
    "SecurityFinding",
    "SecurityReview",
    "Severity",
    "FindingConfidence",
    "CodeChunk",
    "ChunkMetadata",
    "StructuralMetadata",
    "FunctionInfo",
    "ImportInfo",
    "CallInfo",
    "ClassInfo",
    "PromptContext",
    "PromptTemplate",
]