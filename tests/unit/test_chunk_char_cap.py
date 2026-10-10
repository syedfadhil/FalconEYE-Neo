"""Unit tests for the per-chunk character cap in the indexer.

Coverage:
  * _char_capped_end: fits, cut, single over-long line
  * Chunks under the cap are exactly as before (no-cap == cap output)
  * The cap is respected; every line is still covered; walk advances
  * Overlap of capped chunks is at most 20% of their lines
  * A single over-long line becomes its own chunk
  * Adaptive chunking + cap on a large dense file
  * _process_file sends capped chunks to the embedder
  * Config default and bounds
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import ValidationError

from falconeye.application.commands.index_codebase import (
    DEFAULT_MAX_CHUNK_CHARS,
    IndexCodebaseCommand,
    IndexCodebaseHandler,
    _char_capped_end,
)
from falconeye.domain.models.codebase import Codebase
from falconeye.domain.services.checksum_service import ChecksumService
from falconeye.infrastructure.config.config_models import ChunkingConfig


class _Tokens:
    def count_tokens(self, text):
        return len(text) // 4


@pytest.fixture
def chunk():
    handler = IndexCodebaseHandler.__new__(IndexCodebaseHandler)
    handler.llm_service = _Tokens()
    handler.logger = logging.getLogger("test.chunk_char_cap")
    return handler._chunk_content


def _source(n_lines, width):
    return "".join(f"{i:06d} " + "x" * (width - 8) + "\n" for i in range(n_lines))


def _spans(chunks):
    return [(c.metadata.start_line, c.metadata.end_line) for c in chunks]


def _covered(chunks):
    lines = set()
    for c in chunks:
        lines.update(range(c.metadata.start_line, c.metadata.end_line + 1))
    return lines


# ---------------------------------------------------------------
# _char_capped_end
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("lines, cap, expected", [
    (["aaaa\n"] * 10, 1000, 10),   # everything fits
    (["aaaa\n"] * 10, 12, 2),      # 5+5 fits, a third line would be 15
    (["a" * 50 + "\n", "b\n"], 10, 1),  # single over-long line: keep it alone
])
def test_char_capped_end(lines, cap, expected):
    assert _char_capped_end(lines, 0, len(lines), cap) == expected


# ---------------------------------------------------------------
# _chunk_content with max_chars
# ---------------------------------------------------------------


@pytest.mark.unit
def test_chunks_under_the_cap_are_unchanged(chunk):
    """Short lines never reach the cap, so the output is identical."""
    content = _source(200, 10)
    before = chunk(content, "f.py", "python", chunk_size=50, overlap=10)
    after = chunk(content, "f.py", "python", chunk_size=50, overlap=10,
                  max_chars=DEFAULT_MAX_CHUNK_CHARS)
    assert _spans(after) == _spans(before)
    assert [c.content for c in after] == [c.content for c in before]


@pytest.mark.unit
def test_cap_is_respected_and_every_line_is_covered(chunk):
    content = _source(300, 100)  # 50 lines = 5,000 chars > 1,200 cap
    chunks = chunk(content, "f.py", "python", chunk_size=50, overlap=10, max_chars=1200)
    assert all(len(c.content) <= 1200 for c in chunks)
    assert _covered(chunks) == set(range(1, 301))
    starts = [c.metadata.start_line for c in chunks]
    assert starts == sorted(set(starts))  # strictly advancing


@pytest.mark.unit
def test_capped_chunks_overlap_by_at_most_a_fifth(chunk):
    content = _source(300, 100)  # 12 lines per capped chunk
    chunks = chunk(content, "f.py", "python", chunk_size=50, overlap=10, max_chars=1200)
    for a, b in zip(chunks, chunks[1:]):
        a_lines = a.metadata.end_line - a.metadata.start_line + 1
        overlap = a.metadata.end_line - b.metadata.start_line + 1
        assert 0 <= overlap <= a_lines // 5


@pytest.mark.unit
def test_single_over_long_line_is_its_own_chunk(chunk):
    content = "short\n" + "m" * 5000 + "\n" + "tail\n"
    chunks = chunk(content, "f.js", "javascript", chunk_size=50, overlap=10, max_chars=1000)
    assert _spans(chunks)[:3] == [(1, 1), (2, 2), (3, 3)]
    assert len(chunks[1].content) == 5001  # kept whole, not split mid-line


@pytest.mark.unit
def test_adaptive_chunking_with_cap_on_dense_file(chunk):
    """The real failure shape: >1000 lines -> 200-line adaptive windows
    of ~9,000 chars. Capped, no chunk exceeds the limit."""
    content = _source(20_000, 45)
    uncapped = chunk(content, "s.ts", "typescript", chunk_size=50, overlap=10)
    capped = chunk(content, "s.ts", "typescript", chunk_size=50, overlap=10,
                   max_chars=DEFAULT_MAX_CHUNK_CHARS)
    assert max(len(c.content) for c in uncapped) > 8000
    assert max(len(c.content) for c in capped) <= DEFAULT_MAX_CHUNK_CHARS
    assert _covered(capped) == set(range(1, 20_001))
    assert all(c.metadata.total_chunks == len(capped) for c in capped)


# ---------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.texts = []

    async def generate_embeddings_batch(self, texts):
        self.texts.extend(texts)
        return [[0.0] for _ in texts]

    def count_tokens(self, text):
        return len(text) // 4


class _Sink:
    async def store_chunks(self, chunks, collection="code"):
        pass

    async def store_metadata(self, metadata):
        pass

    def save_file(self, metadata):
        pass

    def analyze_file(self, file_path, content):
        return object()


@pytest.mark.unit
def test_process_file_embeds_capped_chunks(tmp_path):
    src = tmp_path / "dense.ts"
    src.write_text(_source(400, 120), encoding="utf-8")
    llm, sink = _Recorder(), _Sink()
    handler = IndexCodebaseHandler(
        vector_store=sink, metadata_repo=sink, llm_service=llm,
        language_detector=None, ast_analyzer=sink, project_identifier=None,
        checksum_service=ChecksumService(), index_registry=sink,
    )
    handler.logger = logging.getLogger("test.chunk_char_cap")
    command = IndexCodebaseCommand(codebase_path=tmp_path, chunk_size=50, chunk_overlap=10,
                                   max_chunk_chars=2000)
    codebase = Codebase.create(root_path=tmp_path, language="typescript", excluded_patterns=[])
    result = asyncio.run(handler._process_file(src, "typescript", command, codebase, "p"))
    assert result is not None
    assert llm.texts and max(len(t) for t in llm.texts) <= 2000


@pytest.mark.unit
def test_max_chunk_chars_config():
    assert ChunkingConfig().max_chunk_chars == DEFAULT_MAX_CHUNK_CHARS == 4500
    from pathlib import Path
    assert IndexCodebaseCommand(codebase_path=Path(".")).max_chunk_chars == 4500
    for bad in (100, 200_000):
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_chars=bad)
