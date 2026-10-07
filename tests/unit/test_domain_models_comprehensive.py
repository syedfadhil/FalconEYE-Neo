"""Comprehensive unit tests for domain model value objects and aggregates."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID

import pytest

from falconeye.domain.exceptions import InvalidCodebaseError
from falconeye.domain.models.code_chunk import ChunkMetadata, CodeChunk
from falconeye.domain.models.codebase import CodeFile, Codebase
from falconeye.domain.models.document import Document, DocumentChunk, DocumentMetadata
from falconeye.domain.models.prompt import PromptContext
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    SecurityReview,
    Severity,
)


def make_finding(severity=Severity.HIGH, confidence=FindingConfidence.HIGH, **overrides):
    data = dict(
        issue="Issue",
        reasoning="Reasoning",
        mitigation="Mitigation",
        severity=severity,
        confidence=confidence,
        file_path="src/app.py",
        code_snippet="print('x')",
        line_start=1,
        line_end=2,
        cwe_id="CWE-79",
        tags=["xss"],
    )
    data.update(overrides)
    return SecurityFinding.create(**data)


def test_security_finding_create_sets_uuid_defaults_and_serializes():
    finding = make_finding(tags=None, line_start=None, line_end=None, cwe_id=None)

    assert isinstance(finding.id, UUID)
    assert finding.tags == []
    assert finding.to_dict()["severity"] == "high"
    assert finding.to_dict()["confidence"] == "high"
    assert finding.to_dict()["line_start"] is None


@pytest.mark.parametrize("severity", list(Severity))
def test_security_finding_accepts_each_severity(severity):
    assert make_finding(severity=severity).severity is severity


@pytest.mark.parametrize("confidence", list(FindingConfidence))
def test_security_finding_accepts_each_confidence(confidence):
    assert make_finding(confidence=confidence).confidence is confidence


@pytest.mark.parametrize("bad", ["urgent", "CRITICAL", 5, None])
def test_invalid_severity_enum_values_are_rejected_by_enum_constructor(bad):
    with pytest.raises((ValueError, TypeError)):
        Severity(bad)


@pytest.mark.parametrize("bad", ["certain", "HIGH", 0, None])
def test_invalid_confidence_enum_values_are_rejected_by_enum_constructor(bad):
    with pytest.raises((ValueError, TypeError)):
        FindingConfidence(bad)


def test_security_finding_is_frozen_and_value_equal_but_unhashable_due_to_tags():
    finding = make_finding()
    same = SecurityFinding(**finding.__dict__)

    assert finding == same
    with pytest.raises(FrozenInstanceError):
        finding.issue = "changed"
    with pytest.raises(TypeError):
        hash(finding)


def test_security_review_empty_collection_counts_and_completion():
    review = SecurityReview.create("/repo", "python")

    assert review.findings == []
    assert review.get_critical_count() == 0
    assert review.get_high_count() == 0
    assert review.get_medium_count() == 0
    assert review.get_low_count() == 0
    assert review.get_all_languages() == ["python"]

    review.complete()
    assert review.completed_at is not None
    assert review.to_dict()["total_findings"] == 0


def test_security_review_adds_findings_counts_and_detects_languages_by_frequency():
    review = SecurityReview.create("/repo", "python")
    review.add_finding(make_finding(severity=Severity.CRITICAL, file_path="a.py"))
    review.add_finding(make_finding(severity=Severity.HIGH, file_path="b.js"))
    review.add_finding(make_finding(severity=Severity.LOW, file_path="c.js"))

    assert review.get_critical_count() == 1
    assert review.get_high_count() == 1
    assert review.get_low_count() == 1
    assert review.get_findings_by_severity(Severity.INFO) == []
    assert review.get_all_languages() == ["javascript", "python"]


def test_code_chunk_metadata_and_embedding_are_immutable_value_objects():
    metadata = ChunkMetadata(
        file_path="src/app.py",
        language="python",
        start_line=1,
        end_line=3,
        chunk_index=0,
        total_chunks=1,
        has_functions=True,
        has_imports=True,
        function_names=["main"],
    )
    chunk = CodeChunk.create("def main(): pass", metadata, token_count=4)
    embedded = chunk.with_embedding([0.1, 0.2])

    assert chunk.embedding is None
    assert embedded.id == chunk.id
    assert embedded.embedding == [0.1, 0.2]
    assert chunk.to_dict()["metadata"]["function_names"] == ["main"]
    with pytest.raises(FrozenInstanceError):
        chunk.content = "changed"


def test_code_file_create_handles_empty_unicode_and_extension():
    empty = CodeFile.create(Path("empty.py"), "empty.py", "", "python")
    unicode_file = CodeFile.create(Path("emoji.py"), "emoji.py", "print('🦅')\n", "python")

    assert empty.size_bytes == 0
    assert empty.line_count == 0
    assert unicode_file.size_bytes > len(unicode_file.content)
    assert unicode_file.line_count == 1
    assert unicode_file.extension == ".py"


def test_codebase_validates_root_and_aggregates_files(tmp_path):
    codebase = Codebase.create(tmp_path, "python", excluded_patterns=[".venv"])
    py = CodeFile.create(tmp_path / "a.py", "a.py", "a\nb\n", "python")
    js = CodeFile.create(tmp_path / "b.js", "b.js", "x\n", "javascript")

    codebase.add_file(py)
    codebase.add_file(js)

    assert codebase.total_files == 2
    assert codebase.total_lines == 3
    assert codebase.total_size_bytes == py.size_bytes + js.size_bytes
    assert codebase.all_languages == ["python", "javascript"]
    assert codebase.excluded_patterns == [".venv"]


def test_codebase_rejects_missing_path_and_file_path(tmp_path):
    with pytest.raises(InvalidCodebaseError):
        Codebase.create(tmp_path / "missing", "python")
    file_path = tmp_path / "notdir.py"
    file_path.write_text("pass")
    with pytest.raises(InvalidCodebaseError):
        Codebase.create(file_path, "python")


def test_document_extracts_title_sections_keywords_and_chunks(tmp_path):
    content = "# API Guide\nSecurity architecture\n## Authentication\nConfiguration"
    doc = Document.create(tmp_path / "README.md", "README.md", content, "readme")
    chunk = DocumentChunk.create(
        content="Security architecture",
        metadata=doc.metadata,
        start_char=0,
        end_char=21,
        chunk_index=0,
        total_chunks=1,
    )
    embedded = chunk.with_embedding([1.0])
    doc.add_chunk(embedded)

    assert doc.metadata.title == "API Guide"
    assert doc.metadata.sections == ["API Guide", "Authentication"]
    assert {"api", "security", "architecture", "authentication", "configuration"}.issubset(doc.metadata.keywords)
    assert doc.total_chunks == 1
    assert embedded.to_dict()["embedding"] == [1.0]


def test_document_without_heading_has_empty_title_sections_and_keywords(tmp_path):
    doc = Document.create(tmp_path / "notes.txt", "notes.txt", "plain text", "notes")

    assert doc.metadata.title is None
    assert doc.metadata.sections == []
    assert doc.metadata.keywords == []


def test_document_metadata_to_dict_supports_empty_collections():
    metadata = DocumentMetadata(file_path="x.md", document_type="readme")
    assert metadata.to_dict() == {
        "file_path": "x.md",
        "document_type": "readme",
        "title": None,
        "sections": [],
        "keywords": [],
    }


def test_prompt_context_to_prompt_dict_omits_empty_optional_fields():
    ctx = PromptContext(file_path="a.py", code_snippet="pass", language="python")

    assert ctx.to_prompt_dict() == {
        "file_path": "a.py",
        "code_snippet": "pass",
        "language": "python",
        "analysis_type": "review",
    }


def test_prompt_context_format_includes_metadata_related_context_and_truncates():
    ctx = PromptContext(
        file_path="a.py",
        code_snippet="\n".join(f"line{i}" for i in range(1, 7)),
        language="python",
        structural_metadata={"functions": ["f"], "classes": [], "imports": ["os"], "calls": ["f"], "control_flow": "if", "data_flows": "x->y"},
        related_code="helper()",
        related_docs="security docs",
        original_file="old code",
    )

    prompt = ctx.format_for_ai(max_code_lines=4)

    assert "FILE: a.py" in prompt
    assert "... [Truncated 2 lines" in prompt
    assert "STRUCTURAL CONTEXT" in prompt
    assert "RELATED CODE" in prompt
    assert "REFERENCE CONTEXT" in prompt
    assert "ORIGINAL FILE" in prompt


def test_prompt_context_enrichment_returns_raw_prompt():
    ctx = PromptContext("x", "raw prompt", "python", analysis_type="enrichment")
    assert ctx.format_for_ai() == "raw prompt"
