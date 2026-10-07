"""LLM service interface (Port)."""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, AsyncIterator
from ..models.prompt import PromptContext


class LLMService(ABC):
    """
    Port for LLM operations.

    This is the PRIMARY interface for AI-powered analysis.
    ALL security findings come from the LLM, NOT from pattern matching.

    Implementations provide different LLM backends (Ollama, etc.)
    """

    @abstractmethod
    async def analyze_code_security(
        self,
        context: PromptContext,
        system_prompt: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Analyze code for security vulnerabilities using AI.

        This is the core method that performs AI-powered security analysis.
        NO pattern matching - pure LLM reasoning.

        Args:
            context: Full context for AI analysis
            system_prompt: System instructions for the AI
            stream_callback: Optional callback to receive streaming tokens

        Returns:
            Raw AI response (usually JSON with findings)
        """
        pass

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch operation).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def validate_findings(
        self,
        code_snippet: str,
        findings: str,
        context: str,
    ) -> str:
        """
        Use AI to validate security findings and remove false positives.

        The AI evaluates whether findings are genuine security issues.
        NO pattern matching involved.

        Args:
            code_snippet: Original code
            findings: Initial findings from analysis
            context: Additional context

        Returns:
            Validated findings (filtered by AI)
        """
        pass

    @abstractmethod
    async def summarize_findings(
        self,
        findings: List[str],
    ) -> str:
        """
        Use AI to summarize multiple findings.

        Args:
            findings: List of finding descriptions

        Returns:
            Summary of findings
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text for the current model.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if LLM service is available.

        Returns:
            True if service is healthy
        """
        pass

    async def chat_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[dict] = None,
    ) -> str:
        """
        Generic chat call with optional structured-output enforcement.

        Used by services that need one-off completions with a specific
        response shape (e.g. the two-pass verifier). Backends that
        support structured output (LiteLLM proxy in front of vLLM /
        SGLang, OpenAI, Anthropic) SHOULD honor `response_schema` and
        constrain generation to match it. Backends that don't (Ollama's
        current OpenAI-compat surface) should fall back to sending the
        request without a schema and rely on the caller having
        described the schema in the prompt.

        This is NOT abstract -- backends without a specialized
        implementation get the default below, which just runs the
        message pair through analyze_code_security() with a stub
        PromptContext. That works but has no structured-output
        enforcement.

        Args:
            system_prompt: Instructions for the model.
            user_prompt: The user turn.
            response_schema: Optional JSON schema. When provided AND the
                backend supports it, the model is constrained to emit
                only valid JSON matching this schema.

        Returns:
            Raw response text. The caller is responsible for JSON
            parsing.
        """
        # Default fallback: create a stub PromptContext so we can
        # reuse analyze_code_security(). Any backend that wants real
        # schema enforcement should override this method.
        stub_context = PromptContext(
            file_path="<verifier>",
            code_snippet=user_prompt,
            language="",
        )
        return await self.analyze_code_security(
            context=stub_context,
            system_prompt=system_prompt,
        )