"""
LiteLLM adapter — talks to a LiteLLM proxy (or any OpenAI-compatible
endpoint) using the OpenAI HTTP API.

Motivation:
-----------
The Ollama adapter is tied to a single local Ollama server. Real
deployments frequently want a beefier remote inference stack — a
LiteLLM proxy in front of vLLM/SGLang/TGI running on multi-GPU boxes
gives you speculative decoding, tensor parallelism, per-model
autoscaling, and OpenAI-compatible routing across many backends.

LiteLLM proxies expose:
    POST /v1/chat/completions   (OpenAI Chat Completions API)
    POST /v1/embeddings         (OpenAI Embeddings API)

This adapter targets those. It supports:
- Separate host+model for chat vs embeddings (LiteLLM lets you serve
  them on different ports/GPUs; the user's setup uses ports 8200 for
  analysis and 8300 for embeddings, for example).
- Streaming and non-streaming chat completions.
- Batched embedding requests (the OpenAI Embeddings API accepts an
  array of inputs in a single request — the same round-trip as
  Ollama's batch, but native to OpenAI protocol).
- API keys (optional; LiteLLM often protects with a virtual key).
- Retry + circuit breaker (same primitives as OllamaLLMAdapter).

We deliberately do NOT depend on the litellm Python package. The
proxy speaks plain OpenAI protocol, and using the openai SDK (or
httpx) directly keeps our dependency tree small and avoids version
skew with LiteLLM's internal SDK.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional

import httpx

from ...domain.services.llm_service import LLMService
from ...domain.models.prompt import PromptContext
from ..logging import FalconEyeLogger, logging_context
from ..resilience import (
    retry_with_backoff,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
)


class LiteLLMAdapter(LLMService):
    """
    LiteLLM (OpenAI-compatible) LLM adapter.

    All methods have the same semantics as OllamaLLMAdapter — this is a
    drop-in replacement selected via config.llm.provider = "litellm".
    """

    # Same char-based truncation for embeddings as OllamaLLMAdapter — the
    # embeddinggemma model has a ~2K token context; going over just wastes
    # a round trip that will 400.
    _MAX_EMBEDDING_CHARS: int = 8000

    def __init__(
        self,
        # Chat endpoint (analysis)
        chat_base_url: str = "http://localhost:8200/v1",
        chat_model: str = "gemma-4-31b-it-heretic",
        chat_api_key: Optional[str] = None,
        # Embedding endpoint (can be same or different from chat)
        embedding_base_url: Optional[str] = None,   # defaults to chat_base_url if None
        embedding_model: str = "embeddinggemma-300m",
        embedding_api_key: Optional[str] = None,
        # Sampling / limits
        chat_max_tokens: int = 32768,
        temperature: float = 0.0,
        max_response_tokens: int = 8192,
        # HTTP behavior
        request_timeout: float = 600.0,
        # Resilience
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        # Structured output enforcement (vLLM guided JSON / OpenAI
        # response_format). When True, every chat request sends a
        # response_format spec that forces the model to emit valid
        # JSON matching the finding schema. Works with any OpenAI-
        # compatible backend that supports response_format, notably
        # vLLM >= 0.6 which powers most LiteLLM deployments.
        guided_json: bool = True,
    ):
        self.chat_base_url = chat_base_url.rstrip("/")
        self.chat_model = chat_model
        self.chat_api_key = chat_api_key or ""

        self.embedding_base_url = (embedding_base_url or chat_base_url).rstrip("/")
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key or self.chat_api_key

        self.chat_max_tokens = chat_max_tokens
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.request_timeout = request_timeout
        self.guided_json = guided_json

        self.logger = FalconEyeLogger.get_instance()

        # HTTP clients — separate per endpoint so different hosts / keys work.
        # Use httpx.AsyncClient directly rather than the openai SDK: keeps
        # dependencies small and gives us native async everywhere.
        self._chat_client = httpx.AsyncClient(
            base_url=self.chat_base_url,
            timeout=httpx.Timeout(request_timeout, connect=30.0),
            headers=self._auth_header(self.chat_api_key),
            verify=False,
        )
        self._embedding_client = httpx.AsyncClient(
            base_url=self.embedding_base_url,
            timeout=httpx.Timeout(request_timeout, connect=30.0),
            headers=self._auth_header(self.embedding_api_key),
            verify=False,
        )

        # Resilience (same defaults as OllamaLLMAdapter for consistency)
        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=0.1,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                httpx.HTTPError,
            ),
        )
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
            exclude_exceptions=(ValueError, TypeError),
        )
        self.circuit_breaker = CircuitBreaker(
            name="litellm_llm",
            config=cb_config,
        )

    # ---------- LLMService interface ----------

    async def analyze_code_security(
        self,
        context: PromptContext,
        system_prompt: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Same contract as OllamaLLMAdapter.analyze_code_security.
        """
        with logging_context(operation="llm_analysis"):
            start_time = time.time()
            user_prompt = context.format_for_ai()
            prompt_length = len(user_prompt)

            self.logger.info(
                "Starting security analysis",
                extra={
                    "backend": "litellm",
                    "model": self.chat_model,
                    "prompt_length": prompt_length,
                    "temperature": self.temperature,
                    "streaming": stream_callback is not None,
                },
            )

            try:
                if stream_callback:
                    response = await self._chat_stream(
                        system_prompt, user_prompt, stream_callback
                    )
                else:
                    response = await self._chat(system_prompt, user_prompt)

                duration = time.time() - start_time
                self.logger.info(
                    "Security analysis completed",
                    extra={
                        "duration_seconds": round(duration, 2),
                        "prompt_length": prompt_length,
                        "response_length": len(response) if response else 0,
                        "model": self.chat_model,
                    },
                )
                return response
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(
                    "Security analysis failed",
                    exc_info=True,
                    extra={
                        "duration_seconds": round(duration, 2),
                        "error_type": type(e).__name__,
                        "model": self.chat_model,
                        "base_url": self.chat_base_url,
                    },
                )
                raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate one embedding. Uses the batch endpoint under the hood
        with a single-element input, so this and generate_embeddings_batch
        share code paths.
        """
        text = self._truncate_embedding(text)
        vectors = await self._embeddings([text])
        return vectors[0]

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Batch embed. The OpenAI Embeddings API natively accepts a list
        of inputs, so this is one round-trip regardless of batch size
        (subject to the model's max_batch_size, which for LiteLLM /
        vLLM is typically 256+).
        """
        if not texts:
            return []
        prepared = [self._truncate_embedding(t) for t in texts]
        return await self._embeddings(prepared)

    async def validate_findings(
        self, code_snippet: str, findings: str, context: str
    ) -> str:
        system_prompt = (
            "You are a security expert validating security findings.\n"
            "Review each finding carefully and determine if it's a "
            "genuine security issue.\n\n"
            "Remove false positives by checking:\n"
            "1. Is the vulnerability actually present in the code?\n"
            "2. Are there mitigations already in place?\n"
            "3. Is the severity assessment accurate?\n"
            "4. Is the reasoning sound?\n\n"
            "Return only the VALID findings in the same JSON format.\n"
            'If all findings are false positives, return: {"reviews": []}\n'
        )
        user_prompt = (
            f"CODE:\n{code_snippet}\n\n"
            f"INITIAL FINDINGS:\n{findings}\n\n"
            f"ADDITIONAL CONTEXT:\n{context}\n\n"
            "Validate these findings and return only genuine security issues."
        )
        return await self._chat(system_prompt, user_prompt)

    async def summarize_findings(self, findings: List[str]) -> str:
        system_prompt = (
            "You are a security expert summarizing security findings.\n"
            "Create a concise but comprehensive summary of all "
            "identified issues."
        )
        user_prompt = "Summarize these security findings:\n\n" + "\n".join(findings)
        return await self._chat(system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def health_check(self) -> bool:
        """
        Ping the LiteLLM proxy's /health endpoint if present, falling
        back to /v1/models on failure. LiteLLM implements both.
        """
        for path in ("/health", "/models"):
            try:
                r = await self._chat_client.get(path)
                if r.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    async def chat_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[dict] = None,
    ) -> str:
        """
        Generic chat call with optional guided_json enforcement.

        Unlike analyze_code_security() which is hardwired to the
        primary security-analysis schema, this method accepts any
        schema and passes it through the standard OpenAI
        response_format field. The primary user of this is the
        two-pass FindingVerifier, which has its own response schema
        (verdict, verifier_reasoning, counterargument).

        The schema is only applied if the adapter's guided_json flag
        is True. If it's False (e.g. the operator disabled structured
        output for troubleshooting), we send the request without
        response_format and hope the prompt is clear enough.
        """
        @self.circuit_breaker.protect
        @retry_with_backoff(self.retry_config)
        async def _do():
            payload = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_response_tokens,
                "stream": False,
            }
            # Apply the caller-supplied schema (not the standard finding
            # schema). Same envelope shape as _apply_response_format.
            if self.guided_json and response_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            r = await self._chat_client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""

        return await _do()

    # ---------- Internals ----------

    @staticmethod
    def _auth_header(api_key: Optional[str]) -> dict:
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
        # LiteLLM proxies without master_key set will accept unauth
        # requests. Send an empty auth header so intermediaries that
        # require the field don't 400 us.
        return {}

    def _truncate_embedding(self, text: str) -> str:
        if len(text) <= self._MAX_EMBEDDING_CHARS:
            return text
        self.logger.warning(
            "Truncating text for embedding",
            extra={
                "original_len": len(text),
                "truncated_len": self._MAX_EMBEDDING_CHARS,
            },
        )
        return text[: self._MAX_EMBEDDING_CHARS]

    # ---------- Structured output schema ----------

    @staticmethod
    def _finding_json_schema() -> dict:
        """
        JSON Schema for the security-review response.

        Sent as `response_format.json_schema` on every chat request when
        `guided_json=True`. vLLM (and any OpenAI-compatible backend that
        implements response_format) constrains the token sampler so the
        model can ONLY emit tokens that produce valid JSON matching
        this schema.

        This is the strongest form of format enforcement available --
        the model cannot skip required fields because the sampler will
        not permit any continuation that closes the object without
        them.

        Notes on the schema:
        - We keep it deliberately permissive on string content (any
          string is fine) and strict only on structure and required
          fields. Being too strict on content (e.g. enum for severity)
          works but tokens more slowly because the sampler has to
          reject a lot of continuations.
        - additionalProperties=False on the finding object stops the
          model from inventing fields we don't parse.
        - Enum on severity is included because it's a small, closed
          set and the sampler handles it cheaply.
        """
        finding = {
            "type": "object",
            "properties": {
                "issue": {"type": "string"},
                "reasoning": {"type": "string"},
                "mitigation": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low", "info"],
                },
                "confidence": {"type": "number"},
                "code_snippet": {"type": "string"},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
                "cwe_id": {"type": "string"},
                # The extended reasoning fields -- REQUIRED so the
                # sampler forces the model to emit them.
                "reasoning_trace": {"type": "string"},
                "evidence_lines": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "assumed_context": {"type": "string"},
                "why_not_false_positive": {"type": "string"},
            },
            "required": [
                "issue",
                "reasoning",
                "mitigation",
                "severity",
                "confidence",
                "code_snippet",
                "line_start",
                "line_end",
                # Required extended fields -- this is what makes
                # guided_json fix the "missing reasoning_trace" bug.
                "reasoning_trace",
                "evidence_lines",
                "assumed_context",
                "why_not_false_positive",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "reviews": {
                    "type": "array",
                    "items": finding,
                },
            },
            "required": ["reviews"],
            "additionalProperties": False,
        }

    def _apply_response_format(self, payload: dict) -> dict:
        """
        Attach response_format to a chat-completions payload if
        guided_json is enabled. Mutates and returns the same dict.

        We use OpenAI's "json_schema" response_format which vLLM
        implements natively (they internally map it to Outlines /
        xgrammar for constrained generation).
        """
        if not self.guided_json:
            return payload
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "security_findings",
                "strict": True,
                "schema": self._finding_json_schema(),
            },
        }
        return payload

    async def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Non-streaming chat completion, wrapped in retry + circuit breaker."""

        @self.circuit_breaker.protect
        @retry_with_backoff(self.retry_config)
        async def _do():
            payload = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_response_tokens,
                "stream": False,
            }
            payload = self._apply_response_format(payload)
            r = await self._chat_client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""

        return await _do()

    async def _chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        stream_callback: Callable[[str], None],
    ) -> str:
        """
        Streaming chat completion. Retry-wrapped as a whole (partial
        streams aren't resumable without server-side session state, so
        on error we just restart the request).
        """

        @self.circuit_breaker.protect
        @retry_with_backoff(self.retry_config)
        async def _do():
            payload = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_response_tokens,
                "stream": True,
            }
            payload = self._apply_response_format(payload)
            full = []
            async with self._chat_client.stream(
                "POST", "/chat/completions", json=payload
            ) as r:
                r.raise_for_status()
                async for raw_line in r.aiter_lines():
                    if not raw_line:
                        continue
                    # OpenAI SSE format: "data: {json}"
                    if not raw_line.startswith("data:"):
                        continue
                    line = raw_line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        import json as _json
                        obj = _json.loads(line)
                    except ValueError:
                        continue
                    try:
                        delta = obj["choices"][0]["delta"]
                    except (KeyError, IndexError, TypeError):
                        continue
                    token = delta.get("content")
                    if token:
                        full.append(token)
                        if stream_callback:
                            try:
                                stream_callback(token)
                            except Exception:
                                # A misbehaving UI callback shouldn't
                                # nuke the whole analysis.
                                pass
            return "".join(full)

        return await _do()

    async def _embeddings(self, inputs: List[str]) -> List[List[float]]:
        """Batched embedding request. Retry + circuit breaker wrapped."""

        @self.circuit_breaker.protect
        @retry_with_backoff(self.retry_config)
        async def _do():
            payload = {"model": self.embedding_model, "input": inputs}
            r = await self._embedding_client.post("/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
            # OpenAI returns results in a possibly-permuted order but with
            # an explicit "index" field. Sort defensively.
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in items]

        return await _do()

    # ---------- Lifecycle ----------

    async def aclose(self) -> None:
        """Close underlying HTTP clients. Safe to call multiple times."""
        for client in (self._chat_client, self._embedding_client):
            try:
                await client.aclose()
            except Exception:
                pass

    def __del__(self) -> None:
        # Best-effort cleanup if aclose() wasn't called. httpx will warn
        # if we let clients get garbage collected without closing.
        try:
            for client in (self._chat_client, self._embedding_client):
                if not client.is_closed:
                    # Can't await in __del__; httpx handles this with a
                    # sync fallback for AsyncClient via close().
                    pass
        except Exception:
            pass
