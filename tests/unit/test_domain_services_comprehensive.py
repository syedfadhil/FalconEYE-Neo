"""Comprehensive unit tests for domain services."""

import asyncio
import hashlib
import sys
import types
from pathlib import Path

import pytest

from falconeye.domain.exceptions import LanguageDetectionError
from falconeye.domain.models.code_chunk import ChunkMetadata, CodeChunk
from falconeye.domain.models.document import DocumentChunk, DocumentMetadata
from falconeye.domain.models.structural import StructuralMetadata
from falconeye.domain.services.checksum_service import ChecksumService
from falconeye.domain.services.context_assembler import ContextAssembler
from falconeye.domain.services.language_detector import LanguageDetector
from falconeye.domain.services.project_identifier import ProjectIdentifier
from falconeye.domain.value_objects.project_metadata import FileMetadata, ProjectType


class FakeMetadataRepo:
    def __init__(self, metadata=None):
        self.metadata = metadata

    async def get_metadata(self, file_path):
        return self.metadata


class FakeVectorStore:
    def __init__(self, code_chunks=None, doc_chunks=None):
        self.code_chunks = code_chunks or []
        self.doc_chunks = doc_chunks or []
        self.code_calls = []
        self.doc_calls = []

    async def search_similar(self, **kwargs):
        self.code_calls.append(kwargs)
        return self.code_chunks

    async def search_similar_documents(self, **kwargs):
        self.doc_calls.append(kwargs)
        return self.doc_chunks


def install_fake_ollama(monkeypatch):
    """Install fake Ollama adapter for ContextAssembler's local lazy import."""
    module = types.ModuleType("falconeye.infrastructure.llm_providers.ollama_adapter")

    class FakeOllamaLLMAdapter:
        async def generate_embedding(self, text):
            return [float(len(text))]

    module.OllamaLLMAdapter = FakeOllamaLLMAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)


def chunk(path, content):
    return CodeChunk.create(
        content,
        ChunkMetadata(path, "python", 1, 1, 0, 1),
        token_count=len(content.split()),
    )


def doc_chunk(path, content):
    return DocumentChunk.create(
        content,
        DocumentMetadata(path, "security_policy"),
        0,
        len(content),
        0,
        1,
    )


@pytest.mark.asyncio
async def test_context_assembler_retrieves_top_k_filters_current_file_and_docs(monkeypatch):
    install_fake_ollama(monkeypatch)
    metadata = StructuralMetadata("src/app.py", "python")
    vector = FakeVectorStore(
        code_chunks=[
            chunk("src/app.py", "same file"),
            chunk("src/auth.py", "auth helper"),
            chunk("src/db.py", "db helper"),
            chunk("src/extra.py", "extra helper"),
        ],
        doc_chunks=[doc_chunk("SECURITY.md", "Use parameterized queries")],
    )
    assembler = ContextAssembler(vector, FakeMetadataRepo(metadata))

    ctx = await assembler.assemble_context(
        "src/app.py", "query code", "python", top_k_similar=2, top_k_docs=1
    )

    assert ctx.structural_metadata["file_path"] == "src/app.py"
    assert "src/auth.py" in ctx.related_code
    assert "src/db.py" in ctx.related_code
    assert "src/extra.py" not in ctx.related_code
    assert "src/app.py" not in ctx.related_code
    assert "Use parameterized queries" in ctx.related_docs
    assert vector.code_calls[0]["top_k"] >= 2
    assert vector.code_calls[0]["top_k"] <= 10
    assert vector.doc_calls[0]["top_k"] == 1


@pytest.mark.asyncio
async def test_context_assembler_deduplicates_related_code_by_file_and_content(monkeypatch):
    install_fake_ollama(monkeypatch)
    duplicate_a = chunk("src/auth.py", "same helper")
    duplicate_b = chunk("src/auth.py", "same helper")
    vector = FakeVectorStore(code_chunks=[duplicate_a, duplicate_b, chunk("src/db.py", "db")])
    assembler = ContextAssembler(vector, FakeMetadataRepo(None))

    ctx = await assembler.assemble_context("src/app.py", "query", "python", top_k_similar=5)

    assert ctx.related_code.count("same helper") == 1
    assert "db" in ctx.related_code


@pytest.mark.asyncio
async def test_context_assembler_empty_results_produce_minimal_prompt_context(monkeypatch):
    install_fake_ollama(monkeypatch)
    assembler = ContextAssembler(FakeVectorStore(), FakeMetadataRepo(None))

    ctx = await assembler.assemble_context("src/app.py", "query", "python")

    assert ctx.related_code is None
    assert ctx.related_docs is None
    assert ctx.structural_metadata is None


@pytest.mark.asyncio
async def test_context_assembler_tolerates_repository_errors(monkeypatch):
    install_fake_ollama(monkeypatch)

    class BrokenVector(FakeVectorStore):
        async def search_similar(self, **kwargs):
            raise RuntimeError("boom")

        async def search_similar_documents(self, **kwargs):
            raise RuntimeError("boom")

    class BrokenMetadata:
        async def get_metadata(self, file_path):
            raise RuntimeError("boom")

    ctx = await ContextAssembler(BrokenVector(), BrokenMetadata()).assemble_context("a.py", "x", "python")

    assert ctx.related_code is None
    assert ctx.related_docs is None
    assert ctx.structural_metadata is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.c", "c"),
        ("a.cpp", "cpp"),
        ("a.py", "python"),
        ("a.rs", "rust"),
        ("a.go", "go"),
        ("a.php", "php"),
        ("a.java", "java"),
        ("a.dart", "dart"),
        ("a.js", "javascript"),
        ("a.ts", "typescript"),
        ("a.rb", "ruby"),
        ("a.cs", "csharp"),
    ],
)
def test_language_detector_detects_supported_languages_by_extension(tmp_path, filename, expected):
    path = tmp_path / filename
    path.write_text("// sample")
    assert LanguageDetector().detect_language(path) == expected


@pytest.mark.parametrize(
    ("filename", "shebang", "expected"),
    [
        ("script", "#!/usr/bin/env python3", "python"),
        ("server", "#!/usr/bin/node", "javascript"),
        ("tool", "#!/usr/bin/env ruby", "ruby"),
        ("index", "#!/usr/bin/env php", "php"),
    ],
)
def test_language_detector_detects_supported_shebangs_without_extension(tmp_path, filename, shebang, expected):
    path = tmp_path / filename
    path.write_text(f"{shebang}\nprint('x')\n")
    assert LanguageDetector().detect_language(path) == expected


def test_language_detector_shebang_matches_executable_name_not_substring(tmp_path):
    path = tmp_path / "script"
    path.write_text("#!/usr/bin/notnode\nconsole.log('x')\n")

    with pytest.raises(LanguageDetectionError):
        LanguageDetector().detect_language(path)


def test_language_detector_rejects_unsupported_file_and_empty_directory(tmp_path):
    unsupported = tmp_path / "README.md"
    unsupported.write_text("# docs")
    with pytest.raises(LanguageDetectionError):
        LanguageDetector().detect_language(unsupported)
    with pytest.raises(LanguageDetectionError):
        LanguageDetector().detect_language(tmp_path / "empty")


def test_language_detector_directory_ignores_skipped_dirs_and_uses_mixed_heuristics(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("x")
    (tmp_path / "a.ts").write_text("x")
    (tmp_path / "b.js").write_text("x")

    assert LanguageDetector().detect_language(tmp_path) == "typescript"


def test_language_detector_force_language_validates_supported_values(tmp_path):
    assert LanguageDetector().detect_language(tmp_path, force_language="python") == "python"
    with pytest.raises(LanguageDetectionError):
        LanguageDetector().detect_language(tmp_path, force_language="brainfuck")


def test_checksum_service_calculates_sha256_and_metadata_snapshot(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"print('hi')\n")
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    service = ChecksumService()

    assert service.calculate_file_checksum(path) == expected
    snapshot = service.get_file_metadata_snapshot(path, Path("a.py"), "proj", "python", "abc")
    assert snapshot.file_checksum == expected
    assert snapshot.file_size == len(b"print('hi')\n")
    assert snapshot.git_commit_hash == "abc"


def test_checksum_service_change_detection_quick_checksum_and_missing_file(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("one")
    service = ChecksumService()
    meta = service.get_file_metadata_snapshot(path, Path("a.py"), "proj", "python")

    assert service.has_file_changed_quick(path, meta) is False
    assert service.has_file_changed_checksum(path, meta) is False
    path.write_text("twox")
    assert service.has_file_changed_quick(path, meta) is True
    assert service.has_file_changed_checksum(path, meta) is True
    path.unlink()
    assert service.has_file_changed_quick(path, meta) is True
    assert service.has_file_changed_checksum(path, meta) is True
    with pytest.raises(FileNotFoundError):
        service.calculate_file_checksum(path)


def test_checksum_service_filters_changed_files_with_and_without_checksum(tmp_path):
    unchanged = tmp_path / "same.py"
    touched = tmp_path / "touched.py"
    new = tmp_path / "new.py"
    for p in [unchanged, touched, new]:
        p.write_text("same")
    service = ChecksumService()
    meta_unchanged = service.get_file_metadata_snapshot(unchanged, Path("same.py"), "proj", "python")
    meta_touched = FileMetadata(
        project_id="proj",
        file_path=touched,
        relative_path=Path("touched.py"),
        language="python",
        file_checksum=service.calculate_file_checksum(touched),
        file_size=touched.stat().st_size,
        file_mtime=touched.stat().st_mtime - 10,
    )

    changed, unchanged_files = service.filter_changed_files_efficient(
        [unchanged, touched, new], {unchanged: meta_unchanged, touched: meta_touched}, use_checksum=True
    )

    assert changed == [new]
    assert unchanged in unchanged_files
    assert touched in unchanged_files


def test_project_identifier_generates_stable_git_id_from_normalized_remote(monkeypatch, tmp_path):
    git_root = tmp_path / "Repo Name"
    git_root.mkdir()
    (git_root / ".git").mkdir()
    ident = ProjectIdentifier()
    monkeypatch.setattr(ident, "_get_git_remote_url", lambda root: "github.com/user/repo")

    project_id, name, project_type, remote = ident.identify_project(git_root)

    assert project_id == "repo_name_" + hashlib.sha256(b"github.com/user/repo").hexdigest()[:8]
    assert name == "Repo Name"
    assert project_type is ProjectType.GIT
    assert remote == "github.com/user/repo"


def test_project_identifier_collision_resistance_for_same_name_different_paths(tmp_path):
    p1 = tmp_path / "a" / "same"
    p2 = tmp_path / "b" / "same"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)
    ident = ProjectIdentifier()

    id1, name1, type1, remote1 = ident.identify_project(p1)
    id2, name2, type2, remote2 = ident.identify_project(p2)

    assert name1 == name2 == "same"
    assert type1 is type2 is ProjectType.NON_GIT
    assert remote1 is remote2 is None
    assert id1 != id2
    assert id1.startswith("same_") and id2.startswith("same_")


def test_project_identifier_explicit_id_and_url_normalization(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    ident = ProjectIdentifier()
    monkeypatch.setattr(ident, "_get_git_remote_url", lambda r: "github.com/user/repo")

    assert ident._normalize_git_url("git@github.com:user/repo.git") == "github.com/user/repo"
    assert ident._normalize_git_url("https://github.com/user/repo.git") == "github.com/user/repo"
    assert ident.identify_project(root, explicit_id="My Project!")[0] == "my_project"
