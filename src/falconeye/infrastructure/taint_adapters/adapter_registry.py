"""
Registry of language taint adapters.

Kept as a very small standalone class so tests and downstream services
depend on a stable indirection rather than importing each concrete
adapter directly. When new languages ship (Python, Java, ...), the DI
container registers them here and everything else stays unchanged.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

from ...domain.services.taint_adapter import TaintLanguageAdapter


class AdapterRegistry:
    """Maps language tag to a concrete ``TaintLanguageAdapter``."""

    def __init__(self) -> None:
        self._by_lang: Dict[str, TaintLanguageAdapter] = {}

    def register(self, adapter: TaintLanguageAdapter) -> None:
        """Add an adapter. Duplicate registrations for the same language
        raise; callers should construct the registry once at DI init."""
        lang = adapter.language
        if not lang:
            raise ValueError(
                f"adapter {adapter!r} has empty `language` attribute; "
                f"every TaintLanguageAdapter must declare its language"
            )
        if lang in self._by_lang:
            raise ValueError(
                f"language {lang!r} already registered "
                f"(existing: {type(self._by_lang[lang]).__name__}, "
                f"new: {type(adapter).__name__})"
            )
        self._by_lang[lang] = adapter

    def get(self, language: str) -> Optional[TaintLanguageAdapter]:
        return self._by_lang.get(language)

    def languages(self) -> Iterator[str]:
        return iter(self._by_lang)

    def __contains__(self, language: str) -> bool:
        return language in self._by_lang

    def __len__(self) -> int:
        return len(self._by_lang)
