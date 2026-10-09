"""Unit tests for resilient embedding during indexing.

Coverage:
  * _embed_texts: bounded batches; a rejected chunk is skipped on its
    own (ValueError- and status_code-style rejections); service failures
    (consecutive non-rejection errors, open breaker) abort the file;
    a short response triggers the per-chunk fallback
  * _process_file: partial files store the embeddable chunks and are
    recorded; all-rejected and service-down files fail, aren't saved to
    the registry, and are recorded
  * LiteLLMAdapter (real HTTP code over httpx.MockTransport): 4xx ->
    EmbeddingRequestError with the server's text, not retried, not
    counted by the breaker; 5xx carries the server's text; the
    embedding breaker is separate from chat; chat errors keep the text
  * CLI summary of partial/failed files; config default and bounds
"""

from __future__ import annotations

import asyncio
import json
import logging
from io import StringIO
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from rich.console import Console

from falconeye.adapters.cli.commands import _print_index_problems
from falconeye.application.commands.index_codebase import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EmbeddingServiceError,
    IndexCodebaseCommand,
    IndexCodebaseHandler,
    IndexRunStats,
)
from falconeye.domain.models.codebase import Codebase
from falconeye.domain.services.checksum_service import ChecksumService
from falconeye.infrastructure.config.config_models import LLMConfig
from falconeye.infrastructure.llm_providers.litellm_adapter import (
    EmbeddingRequestError,
    LiteLLMAdapter,
)
from falconeye.infrastructure.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
    RetryConfig,
)


# ---------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------


class _Embedder:
    """Embeds any text; rejects (ValueError) every request containing BAD.

    ``fail`` makes every call raise that exception instead.
    ``short`` makes multi-text batches return one vector too few.
    """

    def __init__(self, fail=None, short=False):
        self.calls = []
        self._fail = fail
        self._short = short

    async def generate_embeddings_batch(self, texts):
        self.calls.append(list(texts))
        if self._fail is not None:
            raise self._fail
        if any("BAD" in t for t in texts):
            raise ValueError("embedding request rejected (HTTP 400): bad input")
        vectors = [[float(len(t))] for t in texts]
        return vectors[:-1] if self._short and len(texts) > 1 else vectors

    def count_tokens(self, text):
        return len(text) // 4


class _StatusError(Exception):
    """Shaped like ollama.ResponseError: carries status_code."""

    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _Store:
    def __init__(self):
        self.stored = []

    async def store_chunks(self, chunks, collection="code"):
        self.stored.extend(chunks)


class _Registry:
    def __init__(self):
        self.saved = []

    def save_file(self, metadata):
        self.saved.append(metadata)


class _Metadata:
    async def store_metadata(self, metadata):
        pass


class _Ast:
    def analyze_file(self, file_path, content):
        return object()


def _handler(llm):
    handler = IndexCodebaseHandler(
        vector_store=_Store(),
        metadata_repo=_Metadata(),
        llm_service=llm,
        language_detector=None,
        ast_analyzer=_Ast(),
        project_identifier=None,
        checksum_service=ChecksumService(),
        index_registry=_Registry(),
    )
    handler.logger = logging.getLogger("test.embedding_resilience")
    return handler


def _embed(handler, texts, batch_size=4):
    return asyncio.run(handler._embed_texts(texts, batch_size, "f.py"))


# ---------------------------------------------------------------
# _embed_texts
# ---------------------------------------------------------------


@pytest.mark.unit
def test_texts_are_sent_in_bounded_batches():
    llm = _Embedder()
    vectors, errors = _embed(_handler(llm), [f"t{i}" for i in range(70)], batch_size=32)
    assert [len(c) for c in llm.calls] == [32, 32, 6]
    assert errors == [] and all(v is not None for v in vectors)


@pytest.mark.unit
def test_rejected_chunk_is_skipped_alone():
    """A failed batch is retried one text at a time; only BAD is lost."""
    llm = _Embedder()
    texts = ["a", "b", "BAD", "c", "d", "e"]
    vectors, errors = _embed(_handler(llm), texts, batch_size=4)
    assert vectors[2] is None
    assert [v is not None for i, v in enumerate(vectors) if i != 2] == [True] * 5
    assert len(errors) == 1 and "bad input" in errors[0]
    # batch [a,b,BAD,c] failed -> 4 singles; batch [d,e] fine.
    assert [len(c) for c in llm.calls] == [4, 1, 1, 1, 1, 2]


@pytest.mark.unit
def test_many_rejected_chunks_do_not_look_like_an_outage():
    """Consecutive rejections are input problems, not a service outage."""
    vectors, errors = _embed(_handler(_Embedder()), ["BAD1", "BAD2", "BAD3", "BAD4", "ok"])
    assert len(errors) == 4 and vectors[4] is not None


@pytest.mark.unit
def test_status_code_4xx_counts_as_input_rejection():
    llm = _Embedder(fail=_StatusError(400))
    vectors, errors = _embed(_handler(llm), ["a", "b", "c", "d"])
    assert vectors == [None] * 4 and len(errors) == 4


@pytest.mark.unit
@pytest.mark.parametrize("error", [ConnectionError("refused"), _StatusError(503), TimeoutError()])
def test_consecutive_service_errors_abort_the_file(error):
    llm = _Embedder(fail=error)
    with pytest.raises(EmbeddingServiceError, match="3 consecutive"):
        _embed(_handler(llm), [f"t{i}" for i in range(10)])
    assert len(llm.calls) == 1 + 3  # the batch, then three singles


@pytest.mark.unit
def test_open_circuit_breaker_aborts_immediately():
    llm = _Embedder(fail=CircuitBreakerError("open"))
    with pytest.raises(EmbeddingServiceError):
        _embed(_handler(llm), ["a", "b", "c"])
    assert len(llm.calls) == 1


@pytest.mark.unit
def test_short_response_falls_back_to_singles():
    llm = _Embedder(short=True)
    vectors, errors = _embed(_handler(llm), ["a", "b", "c"])
    assert errors == [] and all(v is not None for v in vectors)
    assert [len(c) for c in llm.calls] == [3, 1, 1, 1]


# ---------------------------------------------------------------
# _process_file
# ---------------------------------------------------------------


def _process(tmp_path, llm, lines):
    src = tmp_path / "f.py"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    handler = _handler(llm)
    command = IndexCodebaseCommand(
        codebase_path=tmp_path, chunk_size=2, chunk_overlap=0, embedding_batch_size=2,
    )
    codebase = Codebase.create(root_path=tmp_path, language="python", excluded_patterns=[])
    result = asyncio.run(handler._process_file(src, "python", command, codebase, "proj"))
    return handler, result


@pytest.mark.unit
def test_partial_file_stores_embeddable_chunks(tmp_path):
    handler, result = _process(tmp_path, _Embedder(), ["a = 1", "b = 2", "BAD = 3", "c = 4"])
    assert result is not None
    assert result.chunk_count == 1  # chunk 1 ("a","b") stored, chunk 2 skipped
    assert len(handler.vector_store.stored) == 1
    assert len(handler.index_registry.saved) == 1
    (path, skipped, total, error), = handler.last_run_stats.files_partial
    assert (path, skipped, total) == ("f.py", 1, 2) and "bad input" in error
    assert handler.last_run_stats.files_failed == []


@pytest.mark.unit
def test_all_chunks_rejected_fails_the_file(tmp_path):
    handler, result = _process(tmp_path, _Embedder(), ["BAD = 1", "BAD = 2"])
    assert result is None
    assert handler.index_registry.saved == []
    (path, error), = handler.last_run_stats.files_failed
    assert path == "f.py" and "none of 1 chunks" in error


@pytest.mark.unit
def test_service_down_fails_the_file_for_retry(tmp_path):
    handler, result = _process(
        tmp_path, _Embedder(fail=ConnectionError("refused")), [f"x{i} = {i}" for i in range(8)],
    )
    assert result is None
    assert handler.index_registry.saved == []  # not saved -> retried next run
    assert "EmbeddingServiceError" in handler.last_run_stats.files_failed[0][1]


# ---------------------------------------------------------------
# LiteLLMAdapter over a mock transport
# ---------------------------------------------------------------


def _adapter(embed_handler, chat_handler=None):
    adapter = LiteLLMAdapter(
        chat_base_url="http://chat.test/v1",
        embedding_base_url="http://embed.test/v1",
        retry_config=RetryConfig(
            max_retries=2, initial_delay=0.0, max_delay=0.0, jitter=0.0,
            retryable_exceptions=(ConnectionError, TimeoutError, OSError, httpx.HTTPError),
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=2, success_threshold=1, timeout=60.0,
            exclude_exceptions=(ValueError, TypeError),
        ),
    )
    adapter._embedding_client = httpx.AsyncClient(
        base_url="http://embed.test/v1", transport=httpx.MockTransport(embed_handler),
    )
    if chat_handler is not None:
        adapter._chat_client = httpx.AsyncClient(
            base_url="http://chat.test/v1", transport=httpx.MockTransport(chat_handler),
        )
    return adapter


_OLLAMA_400 = {"error": {"message": 'Post "http://127.0.0.1:60395/tokenize": connection refused'}}


@pytest.mark.unit
def test_embedding_4xx_raises_with_server_text_and_no_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json=_OLLAMA_400)

    adapter = _adapter(handler)
    with pytest.raises(EmbeddingRequestError) as exc:
        asyncio.run(adapter.generate_embeddings_batch(["x"]))
    assert exc.value.status_code == 400
    assert "60395/tokenize" in str(exc.value)
    assert len(calls) == 1  # rejected requests are not resent


@pytest.mark.unit
def test_embedding_4xx_does_not_trip_the_breaker():
    adapter = _adapter(lambda request: httpx.Response(400, json=_OLLAMA_400))
    for _ in range(5):
        with pytest.raises(EmbeddingRequestError):
            asyncio.run(adapter.generate_embeddings_batch(["x"]))
    assert adapter.embedding_circuit_breaker.state == CircuitBreakerState.CLOSED


@pytest.mark.unit
def test_embedding_5xx_keeps_server_text():
    adapter = _adapter(lambda request: httpx.Response(503, text="model is loading"))
    with pytest.raises(httpx.HTTPStatusError, match="server said: model is loading"):
        asyncio.run(adapter.generate_embeddings_batch(["x"]))


@pytest.mark.unit
def test_embedding_breaker_is_separate_from_chat():
    """Embedding trouble during indexing must not block analysis calls."""
    def chat(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = _adapter(lambda request: httpx.Response(503, text="down"), chat)
    for _ in range(3):
        with pytest.raises((httpx.HTTPStatusError, CircuitBreakerError)):
            asyncio.run(adapter.generate_embeddings_batch(["x"]))
    assert adapter.embedding_circuit_breaker.state == CircuitBreakerState.OPEN
    assert adapter.circuit_breaker.state == CircuitBreakerState.CLOSED
    assert asyncio.run(adapter.chat_with_schema("s", "u")) == "ok"


@pytest.mark.unit
def test_chat_errors_keep_server_text():
    def chat(request):
        return httpx.Response(400, json={"error": {"message": "schema not supported"}})

    adapter = _adapter(lambda request: httpx.Response(200, json={"data": []}), chat)
    with pytest.raises(httpx.HTTPStatusError, match="schema not supported"):
        asyncio.run(adapter.chat_with_schema("s", "u", response_schema={"type": "object"}))


@pytest.mark.unit
def test_streaming_chat_errors_keep_server_text():
    def chat(request):
        return httpx.Response(400, json={"error": {"message": "context too long"}})

    adapter = _adapter(lambda request: httpx.Response(200, json={"data": []}), chat)
    with pytest.raises(httpx.HTTPStatusError, match="context too long"):
        asyncio.run(adapter._chat_stream("s", "u", None))


# ---------------------------------------------------------------
# CLI summary and config
# ---------------------------------------------------------------


@pytest.mark.unit
def test_index_summary_lists_partial_and_failed_files():
    stats = IndexRunStats()
    stats.files_partial.append(("libs\\big.stub.ts", 3, 578, "EmbeddingRequestError: [/x] 400"))
    for i in range(6):
        stats.files_failed.append((f"f{i}.ts", "EmbeddingServiceError: down"))
    buf = StringIO()
    _print_index_problems(Console(file=buf, force_terminal=False, width=300), stats)
    out = buf.getvalue()
    assert "1 file(s) indexed partially: 3 chunk(s) could not be embedded" in out
    assert "libs\\big.stub.ts: 3/578 chunks skipped (EmbeddingRequestError: [/x] 400)" in out
    assert "6 file(s) could not be indexed" in out
    assert "... and 1 more" in out


@pytest.mark.unit
def test_index_summary_silent_when_clean():
    buf = StringIO()
    _print_index_problems(Console(file=buf, force_terminal=False), IndexRunStats())
    assert buf.getvalue() == ""


@pytest.mark.unit
def test_embedding_batch_size_config():
    assert LLMConfig().embedding_batch_size == DEFAULT_EMBEDDING_BATCH_SIZE
    assert IndexCodebaseCommand(codebase_path=Path(".")).embedding_batch_size == DEFAULT_EMBEDDING_BATCH_SIZE
    for bad in (0, 513):
        with pytest.raises(ValidationError):
            LLMConfig(embedding_batch_size=bad)
