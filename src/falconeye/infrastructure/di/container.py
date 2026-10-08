"""Dependency injection container for FalconEYE."""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from ..config.config_loader import ConfigLoader
from ..config.config_models import FalconEyeConfig
from ..llm_providers.ollama_adapter import OllamaLLMAdapter
from ...domain.services.llm_service import LLMService
from ..resilience import RetryConfig, CircuitBreakerConfig
from ...domain.services.memory_service import MemoryService
from ..vector_stores.chroma_adapter import ChromaVectorStoreAdapter
from ..persistence.chroma_metadata_repository import ChromaMetadataRepository
from ..registry.chroma_registry_adapter import ChromaIndexRegistryAdapter
from ..ast.ast_analyzer import EnhancedASTAnalyzer
from ..plugins.plugin_registry import PluginRegistry
from ...domain.services.security_analyzer import SecurityAnalyzer
from ...domain.services.context_assembler import ContextAssembler
from ...domain.services.language_detector import LanguageDetector
from ...domain.services.project_identifier import ProjectIdentifier
from ...domain.services.checksum_service import ChecksumService
from ...application.commands.index_codebase import IndexCodebaseHandler
from ...application.commands.review_file import ReviewFileHandler
from ..persistence.checkpoint_service import CheckpointService
from ..taint_index import TaintIndex, load_taint_index
from ..taint_adapters.adapter_registry import AdapterRegistry
from ..taint_adapters.csharp import CSharpTaintAdapter
from ...domain.services.taint_verifier import TaintVerifier
from ...application.services.taint_scan_service import TaintScanService
from ..skills.skill_loader import SkillLoader
from ...application.services.skill_runner_service import SkillRunnerService
from ...application.services.review_confirmation_service import ReviewConfirmationService
from ...domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)


def _resolve_skills_directory(
    *,
    config_value: str,
    config_path: "Optional[str]",
) -> Path:
    """Resolve the effective skills directory.

    Resolution order, highest priority first:

      1. FALCONEYE_SKILLS_DIR environment variable (if set and non-empty).
         Overrides everything. Handy for CI pipelines and portable configs
         where the same config.yaml runs against a different skills library
         on each machine.
      2. The config value (config.skills.directory). Absolute paths (including
         those starting with ~) are used as-is; relative paths resolve against
         the config file's parent directory so ``./skills`` in ``~/proj/config.yaml``
         means ``~/proj/skills``.

    All paths get ``~`` and ``$VAR`` expansion via ``os.path.expanduser`` /
    ``os.path.expandvars`` before being converted to Path.
    """
    env_value = os.environ.get("FALCONEYE_SKILLS_DIR", "").strip()
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value)))

    raw = os.path.expandvars(os.path.expanduser(config_value))
    candidate = Path(raw)
    if not candidate.is_absolute() and config_path is not None:
        candidate = Path(config_path).resolve().parent / candidate
    return candidate


@dataclass
class DIContainer:
    """
    Dependency injection container for FalconEYE.

    Assembles all components with proper dependency injection.
    This container is created once at application startup.
    """

    # Configuration
    config: FalconEyeConfig

    # Infrastructure
    llm_service: LLMService
    vector_store: ChromaVectorStoreAdapter
    metadata_repo: ChromaMetadataRepository
    index_registry: ChromaIndexRegistryAdapter
    ast_analyzer: EnhancedASTAnalyzer
    plugin_registry: PluginRegistry

    # Domain Services
    security_analyzer: SecurityAnalyzer
    context_assembler: ContextAssembler
    language_detector: LanguageDetector
    project_identifier: ProjectIdentifier
    checksum_service: ChecksumService

    # Application Handlers
    index_handler: IndexCodebaseHandler
    review_file_handler: ReviewFileHandler

    # Resumable-scan support
    checkpoint_service: CheckpointService

    # Optional Services
    memory_service: Optional[MemoryService] = None

    # Optional taint analysis services. Both are None when no
    # language in config.languages.per_language has taint enabled --
    # this keeps startup cheap for the default install path.
    taint_index: Optional[TaintIndex] = None
    taint_adapter_registry: Optional[AdapterRegistry] = None
    taint_scan_service: Optional[TaintScanService] = None
    skill_loader: Optional[SkillLoader] = None
    skill_runner_service: Optional[SkillRunnerService] = None
    review_confirmation_service: Optional[ReviewConfirmationService] = None

    @classmethod
    def create(cls, config_path: Optional[str] = None, backend_override: Optional[str] = None, sage_override: bool = False) -> "DIContainer":
        """
        Create and wire all dependencies.

        Args:
            config_path: Optional path to configuration file
            backend_override: Optional LLM backend override
            sage_override: Force-enable SAGE persistent memory

        Returns:
            DIContainer with all dependencies wired
        """
        # Load configuration
        config = ConfigLoader.load(config_path)

        # Apply SAGE override from CLI flag
        if sage_override:
            config.sage.enabled = True

        # Create data directories if they don't exist
        Path(config.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(config.metadata.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(config.index_registry.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(config.output.output_directory).mkdir(parents=True, exist_ok=True)

        # Infrastructure layer - Adapters

        # Create retry config from configuration
        retry_config = RetryConfig(
            max_retries=config.llm.retry.max_retries,
            initial_delay=config.llm.retry.initial_delay,
            max_delay=config.llm.retry.max_delay,
            exponential_base=config.llm.retry.exponential_base,
            jitter=config.llm.retry.jitter,
            retryable_exceptions=(ConnectionError, TimeoutError, OSError)
        )

        # Create circuit breaker config from configuration
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=config.llm.circuit_breaker.failure_threshold,
            success_threshold=config.llm.circuit_breaker.success_threshold,
            timeout=config.llm.circuit_breaker.timeout,
            exclude_exceptions=(ValueError, TypeError)
        )

        # Determine which backend to use
        provider = backend_override or config.llm.provider

        if provider == "mlx":
            # MLX backend for Apple Silicon
            from ..llm_providers.mlx_adapter import MLXLLMAdapter, is_apple_silicon, is_mlx_available

            if not is_apple_silicon():
                raise RuntimeError(
                    "MLX backend requires Apple Silicon (M1/M2/M3/M4). "
                    "Use --backend ollama instead."
                )
            if not is_mlx_available():
                raise RuntimeError(
                    "MLX packages not installed. Install with: pip install falconeye[mlx]"
                )

            llm_service = MLXLLMAdapter(
                model_path=config.llm.mlx.analysis,
                ollama_host=config.llm.base_url,
                embedding_model=config.llm.model.embedding,
                max_tokens=config.llm.mlx.max_tokens,
                circuit_breaker_config=circuit_breaker_config,
            )
        elif provider == "litellm":
            # LiteLLM proxy (or any OpenAI-compatible endpoint).
            # Chat and embeddings can be split across different hosts,
            # matching real deployments where analysis runs on a big
            # multi-GPU node and embeddings run on a smaller node.
            from ..llm_providers.litellm_adapter import LiteLLMAdapter

            llm_service = LiteLLMAdapter(
                chat_base_url=config.llm.litellm.chat_base_url,
                chat_model=config.llm.litellm.chat_model,
                chat_api_key=config.llm.litellm.chat_api_key or None,
                embedding_base_url=(
                    config.llm.litellm.embedding_base_url or None
                ),
                embedding_model=config.llm.litellm.embedding_model,
                embedding_api_key=(
                    config.llm.litellm.embedding_api_key or None
                ),
                chat_max_tokens=config.llm.litellm.max_tokens,
                request_timeout=config.llm.litellm.request_timeout,
                retry_config=retry_config,
                circuit_breaker_config=circuit_breaker_config,
                guided_json=config.llm.litellm.guided_json,
            )
        else:
            llm_service = OllamaLLMAdapter(
                host=config.llm.base_url,
                chat_model=config.llm.model.analysis,
                embedding_model=config.llm.model.embedding,
                retry_config=retry_config,
                circuit_breaker_config=circuit_breaker_config,
            )

        vector_store = ChromaVectorStoreAdapter(
            persist_directory=config.vector_store.persist_directory,
            collection_prefix=config.vector_store.collection_prefix,
        )

        metadata_repo = ChromaMetadataRepository(
            persist_directory=config.metadata.persist_directory,
            collection_name=config.metadata.collection_name,
        )

        index_registry = ChromaIndexRegistryAdapter(
            persist_directory=config.index_registry.persist_directory,
            collection_name=config.index_registry.collection_name,
        )

        ast_analyzer = EnhancedASTAnalyzer()

        plugin_registry = PluginRegistry()
        plugin_registry.load_all_plugins()

        # Two-pass finding verifier (adversarial second-pass review).
        # Gated by config.analysis.two_pass_verification. Optionally
        # uses a separate LLM endpoint if config.analysis.verifier.*
        # is populated -- this supports cross-model verification. If
        # any verifier.* URL field is empty, we reuse the primary
        # llm_service.
        finding_verifier = None
        if config.analysis.two_pass_verification:
            from ..llm_providers.litellm_adapter import LiteLLMAdapter
            from ...domain.services.finding_verifier import FindingVerifier

            verifier_llm = llm_service  # default: same as primary
            v_cfg = config.analysis.verifier
            if provider == "litellm" and v_cfg.chat_base_url:
                # Build a dedicated LLM service for verification. This
                # only makes sense when the primary is also litellm --
                # for Ollama/MLX the split-endpoint concept doesn't
                # apply.
                verifier_llm = LiteLLMAdapter(
                    chat_base_url=v_cfg.chat_base_url,
                    chat_model=v_cfg.chat_model or config.llm.litellm.chat_model,
                    chat_api_key=(
                        v_cfg.chat_api_key
                        or config.llm.litellm.chat_api_key
                        or None
                    ),
                    # Embedding endpoint is not used by the verifier
                    # but LiteLLMAdapter needs one -- reuse primary.
                    embedding_base_url=(
                        config.llm.litellm.embedding_base_url or None
                    ),
                    embedding_model=config.llm.litellm.embedding_model,
                    embedding_api_key=(
                        config.llm.litellm.embedding_api_key or None
                    ),
                    chat_max_tokens=config.llm.litellm.max_tokens,
                    request_timeout=config.llm.litellm.request_timeout,
                    retry_config=retry_config,
                    circuit_breaker_config=circuit_breaker_config,
                    guided_json=config.llm.litellm.guided_json,
                )

            finding_verifier = FindingVerifier(
                llm_service=verifier_llm,
                context_lines=v_cfg.context_lines,
                concurrency=v_cfg.concurrency,
            )

        # Domain services - Business logic
        security_analyzer = SecurityAnalyzer(
            llm_service,
            strict_validation=config.analysis.strict_validation,
            finding_verifier=finding_verifier,
        )
        # Inject llm_service so the assembler doesn't build ad-hoc Ollama
        # clients per RAG call (fixes: two OllamaLLMAdapter() instances
        # per file, and duplicate embedding generation for the same query).
        context_assembler = ContextAssembler(vector_store, metadata_repo, llm_service)
        language_detector = LanguageDetector()
        project_identifier = ProjectIdentifier()
        checksum_service = ChecksumService()

        # Optional: SAGE persistent memory
        # Health is checked lazily on first use via SAGEMemoryAdapter.health_check()
        # to avoid blocking the sync DI factory with a network call.
        memory_service = None
        if config.sage.enabled:
            try:
                from ..memory.sage_adapter import SAGEMemoryAdapter
                from ..logging import FalconEyeLogger

                sage_logger = FalconEyeLogger.get_instance()
                memory_service = SAGEMemoryAdapter(
                    base_url=config.sage.base_url,
                    identity_path=config.sage.identity_path,
                    timeout=config.sage.timeout,
                    store_throttle_seconds=config.sage.store_throttle_seconds,
                )
                sage_logger.info(
                    "SAGE memory adapter initialized (health checked on first use)",
                    extra={"base_url": config.sage.base_url},
                )
            except Exception as e:
                from ..logging import FalconEyeLogger

                sage_logger = FalconEyeLogger.get_instance()
                sage_logger.warning(f"Failed to initialize SAGE memory: {e}")
                memory_service = None

        # Application handlers - Use cases
        index_handler = IndexCodebaseHandler(
            vector_store=vector_store,
            metadata_repo=metadata_repo,
            llm_service=llm_service,
            language_detector=language_detector,
            ast_analyzer=ast_analyzer,
            project_identifier=project_identifier,
            checksum_service=checksum_service,
            index_registry=index_registry,
        )

        review_file_handler = ReviewFileHandler(
            security_analyzer=security_analyzer,
            context_assembler=context_assembler,
            memory_service=memory_service,
            recall_context=config.sage.recall_context,
            store_findings=config.sage.store_findings,
        )

        # Checkpoint dir lives next to the vector store so it moves with
        # the rest of falconeye's on-disk state.
        checkpoint_dir = Path(config.vector_store.persist_directory).parent / "checkpoints"
        checkpoint_service = CheckpointService(checkpoint_dir)

        # Taint analysis wiring. Only constructs when at least one
        # language explicitly opts in via config.languages.per_language.
        # See taint_specs/SCHEMA.md for the spec vocabulary the index
        # loads. The adapter registry is populated only for languages
        # that both (a) opted in AND (b) have a shipped adapter.
        taint_index: Optional[TaintIndex] = None
        taint_adapter_registry: Optional[AdapterRegistry] = None
        taint_enabled_languages = [
            lang
            for lang, settings in config.languages.per_language.items()
            if settings.taint.enabled and lang in config.languages.enabled
        ]
        taint_scan_service: Optional[TaintScanService] = None
        if taint_enabled_languages:
            taint_index = load_taint_index(languages=taint_enabled_languages)
            taint_adapter_registry = AdapterRegistry()
            for lang in taint_enabled_languages:
                if lang == "csharp":
                    taint_adapter_registry.register(CSharpTaintAdapter())
                # Additional adapters register here as they land.

            taint_verifier = TaintVerifier(llm_service=llm_service)
            taint_scan_service = TaintScanService(
                adapter_registry=taint_adapter_registry,
                taint_index=taint_index,
                verifier=taint_verifier,
            )

        skill_loader: Optional[SkillLoader] = None
        if getattr(config, "skills", None) is not None and config.skills.enabled:
            skills_dir = _resolve_skills_directory(
                config_value=config.skills.directory,
                config_path=config_path,
            )
            skill_loader = SkillLoader(directory=skills_dir)


        # Build review confirmation service if we have both a
        # taint adapter registry and taint index available. It
        # reuses the same infrastructure the taint scan uses.
        review_confirmation_service: Optional[ReviewConfirmationService] = None
        if taint_scan_service is not None:
            entry_registry = EntryPointDetectorRegistry()
            entry_registry.register(AspNetCoreEntryPointDetector())
            review_confirmation_service = ReviewConfirmationService(
                adapter_registry=taint_scan_service._adapter_registry,
                entry_point_registry=entry_registry,
                taint_index=taint_scan_service._taint_index,
            )

        # Build the skill runner when we have both an LLM service
        # and a skill loader wired up. The runner is stateless so we
        # construct one per container and reuse it across skill runs.
        skill_runner_service: Optional[SkillRunnerService] = None
        if skill_loader is not None and llm_service is not None:
            skill_runner_service = SkillRunnerService(llm_service=llm_service)

        return cls(
            config=config,
            llm_service=llm_service,
            vector_store=vector_store,
            metadata_repo=metadata_repo,
            index_registry=index_registry,
            ast_analyzer=ast_analyzer,
            plugin_registry=plugin_registry,
            memory_service=memory_service,
            security_analyzer=security_analyzer,
            context_assembler=context_assembler,
            language_detector=language_detector,
            project_identifier=project_identifier,
            checksum_service=checksum_service,
            index_handler=index_handler,
            review_file_handler=review_file_handler,
            checkpoint_service=checkpoint_service,
            taint_index=taint_index,
            taint_adapter_registry=taint_adapter_registry,
            taint_scan_service=taint_scan_service,
            skill_loader=skill_loader,
            skill_runner_service=skill_runner_service,
            review_confirmation_service=review_confirmation_service,
        )

    def get_system_prompt(self, language: str) -> str:
        """
        Get system prompt for a language.

        Appends severity calibration guidelines and the reasoning/
        validation addendum to every prompt so all LLM backends
        produce findings in the same enriched, validation-friendly
        format.

        Args:
            language: Language name

        Returns:
            System prompt string with severity guidelines and
            reasoning/CWE addendum
        """
        from ..plugins.base_plugin import LanguagePlugin
        from ..finding_validation import get_allowed_cwe_list_for_prompt

        severity_guidelines = LanguagePlugin.get_severity_guidelines()
        # Reasoning addendum forces chain-of-thought, evidence_lines,
        # assumed_context, why_not_false_positive, and constrains CWEs
        # to an enumerated list. See LanguagePlugin.get_reasoning_addendum
        # for rationale.
        reasoning_addendum = LanguagePlugin.get_reasoning_addendum(
            allowed_cwe_list=get_allowed_cwe_list_for_prompt(),
        )

        plugin = self.plugin_registry.get_plugin(language)
        if plugin:
            return plugin.get_system_prompt() + severity_guidelines + reasoning_addendum

        # Default generic prompt if no plugin
        default_prompt = """You are a security expert analyzing code for vulnerabilities.
Analyze the provided code and identify any security issues.

Output format (JSON):
{
  "reviews": [
    {
      "issue": "Brief description",
      "reasoning": "Detailed explanation",
      "mitigation": "How to fix",
      "severity": "critical|high|medium|low|info",
      "confidence": 0.9,
      "code_snippet": "Vulnerable code"
    }
  ]
}

If no issues found, return: {"reviews": []}"""
        return default_prompt + severity_guidelines + reasoning_addendum

    def __repr__(self) -> str:
        """String representation."""
        return f"<DIContainer: {len(self.plugin_registry.get_supported_languages())} languages>"