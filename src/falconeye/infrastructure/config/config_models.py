"""Configuration data models using Pydantic."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict


class LLMModelConfig(BaseModel):
    """LLM model configuration."""
    analysis: str = Field(
        default="qwen3-coder:30b",
        description="Model for security analysis"
    )
    embedding: str = Field(
        default="embeddinggemma:300m",
        description="Model for generating embeddings"
    )


class MLXModelConfig(BaseModel):
    """MLX model configuration for Apple Silicon."""
    analysis: str = Field(
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        description="HuggingFace model path for MLX inference"
    )
    max_tokens: int = Field(
        default=32768,
        ge=1024,
        le=131072,
        description="Maximum tokens to generate per response (higher needed for thinking models)"
    )


class LiteLLMConfig(BaseModel):
    """
    LiteLLM proxy configuration.

    LiteLLM speaks OpenAI protocol. Chat and embeddings can be served
    on the same host or split across different hosts / ports — the
    user's deployment example puts analysis on :8200 and embeddings on
    :8300 with different GPU groups behind each.
    """
    chat_base_url: str = Field(
        default="http://localhost:8200/v1",
        description="Base URL for chat/analysis endpoint (must include /v1)"
    )
    chat_model: str = Field(
        default="gemma-4-31b-it-heretic",
        description="Model name as registered in LiteLLM for chat/analysis"
    )
    chat_api_key: str = Field(
        default="",
        description="API key for the chat endpoint (LiteLLM virtual key). Empty for unauth."
    )
    embedding_base_url: str = Field(
        default="",
        description=(
            "Base URL for embeddings endpoint. If empty, chat_base_url is "
            "used. Set this when embeddings run on a different host/port."
        )
    )
    embedding_model: str = Field(
        default="embeddinggemma-300m",
        description="Model name as registered in LiteLLM for embeddings"
    )
    embedding_api_key: str = Field(
        default="",
        description="API key for the embedding endpoint. Empty falls back to chat_api_key."
    )
    max_tokens: int = Field(
        default=32768,
        ge=1024,
        le=1048576,
        description="Chat context window / max input tokens"
    )
    request_timeout: float = Field(
        default=600.0,
        ge=10.0,
        le=3600.0,
        description="HTTP request timeout in seconds"
    )
    guided_json: bool = Field(
        default=True,
        description=(
            "Enforce structured output via OpenAI response_format / vLLM "
            "guided_json. When True, every chat request sends a JSON "
            "schema and the backend constrains token sampling so the "
            "model can ONLY emit valid JSON matching the finding schema "
            "-- including all required reasoning fields. Requires vLLM "
            ">= 0.6, SGLang, or another backend that supports "
            "response_format.json_schema. Set to false if your backend "
            "doesn't support it (Ollama's OpenAI-compat endpoint does "
            "not) or if you're troubleshooting output shape issues."
        )
    )


class RetryConfigModel(BaseModel):
    """Retry logic configuration."""
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    initial_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay in seconds before first retry"
    )
    max_delay: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds between retries"
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=3.0,
        description="Exponential backoff base"
    )
    jitter: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Jitter factor (0.1 = ±10% randomness)"
    )


class CircuitBreakerConfigModel(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of failures before opening circuit"
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of successes to close circuit from half-open"
    )
    timeout: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Seconds to wait before transitioning to half-open"
    )


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = Field(
        default="ollama",
        description="LLM provider (ollama, mlx, litellm)"
    )
    model: LLMModelConfig = Field(default_factory=LLMModelConfig)
    mlx: MLXModelConfig = Field(default_factory=MLXModelConfig)
    litellm: LiteLLMConfig = Field(default_factory=LiteLLMConfig)
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for LLM API (Ollama only; LiteLLM uses litellm.*_base_url)"
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="[DEPRECATED] Use retry.max_retries instead"
    )
    retry: RetryConfigModel = Field(
        default_factory=RetryConfigModel,
        description="Retry logic configuration"
    )
    circuit_breaker: CircuitBreakerConfigModel = Field(
        default_factory=CircuitBreakerConfigModel,
        description="Circuit breaker configuration"
    )

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Ensure provider is valid."""
        valid = ["ollama", "mlx", "litellm"]
        if v not in valid:
            raise ValueError(f"provider must be one of {valid}")
        return v


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    provider: str = Field(
        default="chroma",
        description="Vector store provider (chroma, postgres)"
    )
    persist_directory: str = Field(
        default="./falconeye_data/vectorstore",
        description="Directory for vector store persistence"
    )
    collection_prefix: str = Field(
        default="falconeye",
        description="Prefix for collection names"
    )


class MetadataConfig(BaseModel):
    """Metadata repository configuration."""
    provider: str = Field(
        default="chroma",
        description="Metadata provider (chroma, postgres)"
    )
    persist_directory: str = Field(
        default="./falconeye_data/metadata",
        description="Directory for metadata persistence"
    )
    collection_name: str = Field(
        default="metadata",
        description="Collection name for metadata"
    )


class IndexRegistryConfig(BaseModel):
    """Index registry configuration for project tracking."""
    persist_directory: str = Field(
        default="./falconeye_data/registry",
        description="Directory for registry persistence"
    )
    collection_name: str = Field(
        default="index_registry",
        description="Collection name for index registry"
    )


class ChunkingConfig(BaseModel):
    """Code chunking configuration."""
    default_size: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Default lines per chunk"
    )
    default_overlap: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Default lines of overlap between chunks"
    )
    max_chunk_size: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Maximum lines per chunk"
    )

    @field_validator('default_overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure overlap is less than chunk size."""
        if 'default_size' in info.data and v >= info.data['default_size']:
            raise ValueError("overlap must be less than chunk_size")
        return v


class VerifierConfig(BaseModel):
    """
    Two-pass finding verifier configuration.

    The verifier is an adversarial second-pass reviewer that runs on
    every finding surviving primary analysis + structured validation
    + enrichment. It can confirm, downgrade, reject, or flag findings
    for uncertain evidence.

    By default the verifier uses the same LLM endpoint as primary
    analysis. Setting chat_base_url/chat_model here points it at a
    different endpoint -- useful for cross-model verification with a
    different model family (e.g. Qwen 3.6 Coder verifying findings
    from Gemma 4). Only relevant when provider=litellm.
    """
    concurrency: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Findings verified in parallel"
    )
    context_lines: int = Field(
        default=10,
        ge=0,
        le=100,
        description=(
            "Lines of code above/below the cited lines to include in "
            "the verifier's prompt. Larger values give the verifier "
            "more context but cost more tokens."
        )
    )
    chat_base_url: str = Field(
        default="",
        description=(
            "Optional: separate LLM endpoint for verification (litellm "
            "provider only). Empty falls back to primary chat endpoint."
        )
    )
    chat_model: str = Field(
        default="",
        description=(
            "Optional: separate model name for verification. Empty "
            "falls back to primary chat model."
        )
    )
    chat_api_key: str = Field(
        default="",
        description=(
            "Optional: separate API key for verification endpoint. "
            "Empty falls back to primary chat API key."
        )
    )


class AnalysisConfig(BaseModel):
    """Security analysis configuration."""
    top_k_context: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar chunks for RAG context"
    )
    validate_findings: bool = Field(
        default=False,
        description="Enable AI-based validation to reduce false positives"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of files to process in parallel"
    )
    strict_validation: bool = Field(
        default=True,
        description=(
            "Run structured post-parse validation on every finding: verify "
            "cited line numbers exist, verify code_snippet appears in the "
            "file, restrict cwe_id to an enumerated list, flag suspicious "
            "CVE references. Findings that fail validation are downgraded "
            "and tagged, not dropped. Turn off only if you're seeing valid "
            "findings incorrectly warned/downgraded."
        )
    )
    two_pass_verification: bool = Field(
        default=False,
        description=(
            "Run every finding through an adversarial second-pass verifier "
            "that argues AGAINST the finding to catch false positives. "
            "Approximately doubles LLM cost per file (one verifier call "
            "per finding) but empirically catches 30-50% of FPs that "
            "survive primary analysis. Default off; opt-in until you've "
            "seen how it behaves on your codebase."
        )
    )
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)


class LanguageTaintConfig(BaseModel):
    """Per-language taint analysis settings.

    Nothing runs unless ``enabled`` is set true. Defaults are
    chosen so an operator opting in gets sensible behaviour
    without further tuning:
      - ``max_hops`` bounds the path finder; 5 is generous for
        most codebases and cheap for the LLM verifier
      - ``specs`` empty means "auto-detect frameworks from the
        scanned repo"; a non-empty list pins the spec set
        (useful for reproducible CI runs)
      - ``framework_autodetect`` gates the .csproj / requirements
        scan; set false when ``specs`` is pinned
    """
    enabled: bool = Field(
        default=False,
        description="Enable taint analysis for this language",
    )
    max_hops: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum call-graph edges to traverse from source to sink",
    )
    specs: List[str] = Field(
        default_factory=list,
        description="Explicit spec-file names (empty means auto-detect)",
    )
    framework_autodetect: bool = Field(
        default=True,
        description="Detect frameworks via project files (.csproj, requirements.txt, ...)",
    )

    model_config = ConfigDict(extra="forbid")


class LanguageSettings(BaseModel):
    """Per-language settings block. Currently only holds taint;
    a nested block lets future settings (linter overrides, plugin
    tuning, etc.) coexist without polluting the top-level config."""
    taint: LanguageTaintConfig = Field(default_factory=LanguageTaintConfig)

    model_config = ConfigDict(extra="forbid")


class LanguagesConfig(BaseModel):
    """Language support configuration."""
    enabled: List[str] = Field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "c",
            "cpp",
            "java",
            "dart",
            "php",
        ],
        description="List of enabled languages"
    )
    per_language: Dict[str, LanguageSettings] = Field(
        default_factory=dict,
        description="Per-language settings (taint config, future overrides). Keys are language tags from `enabled`.",
    )

    model_config = ConfigDict(extra="forbid")


class FileDiscoveryConfig(BaseModel):
    """File discovery configuration."""
    default_exclusions: List[str] = Field(
        default_factory=lambda: [
            "*/node_modules/*",
            "*/venv/*",
            "*/virtualenv/*",
            "*/.git/*",
            "*/dist/*",
            "*/build/*",
            "*/__pycache__/*",
            "*/target/*",
            "*.min.js",
            "*.pyc",
        ],
        description="Default file/directory exclusion patterns"
    )


class OutputConfig(BaseModel):
    """Output configuration."""
    default_format: str = Field(
        default="json",
        description="Default output format (console, json, sarif, html)"
    )
    color: bool = Field(
        default=True,
        description="Enable colored console output"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )
    save_to_file: bool = Field(
        default=True,
        description="Save output to file (auto-saves JSON + HTML reports)"
    )
    output_directory: str = Field(
        default="./falconeye_reports",
        description="Directory for saving reports"
    )

    @field_validator('default_format')
    @classmethod
    def validate_format(cls, v):
        """Ensure format is valid."""
        valid_formats = ["console", "json", "sarif", "html"]
        if v not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    file: str = Field(
        default="./falconeye.log",
        description="Log file path"
    )
    console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    rotation: str = Field(
        default="daily",
        description="Log rotation strategy (daily, none)"
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retain logs"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v

    @field_validator('rotation')
    @classmethod
    def validate_rotation(cls, v):
        """Ensure rotation strategy is valid."""
        valid_strategies = ["daily", "none"]
        v = v.lower()
        if v not in valid_strategies:
            raise ValueError(f"rotation must be one of {valid_strategies}")
        return v


class SAGEConfig(BaseModel):
    """SAGE persistent memory configuration."""
    enabled: bool = Field(default=False, description="Enable SAGE memory integration")
    base_url: str = Field(default="http://localhost:8090", description="SAGE API base URL")
    identity_path: Optional[str] = Field(default=None, description="Path to SAGE agent key file")
    timeout: float = Field(default=15.0, ge=1.0, le=120.0, description="API request timeout")
    store_findings: bool = Field(default=True, description="Store scan findings in SAGE")
    recall_context: bool = Field(default=True, description="Recall historical context before analysis")
    store_throttle_seconds: float = Field(
        default=0.5, ge=0.0, le=10.0,
        description="Delay between memory proposals (0 for multi-node deployments)",
    )


class SkillsConfig(BaseModel):
    """Configuration for the skill runner (audit playbooks).

    Skills are external markdown files with YAML frontmatter that
    describe structured audit procedures. When enabled, FalconEYE
    loads them from ``directory`` and exposes them via the ``skill``
    subcommand group. Disabled by default so a stock install has no
    new required directories.
    """
    enabled: bool = Field(
        default=False,
        description="Enable skill loading and the `falconeye skill` subcommand"
    )
    directory: str = Field(
        default="~/.falconeye/skills",
        description=(
            "Path to the skills directory. Absolute paths (including ~) are used as-is; "
            "relative paths resolve against the config file location. The default is a "
            "user-level location so skills stay a global asset shared across projects, "
            "not something duplicated into every analyzed codebase. Overridable via the "
            "FALCONEYE_SKILLS_DIR environment variable or the --skills-dir CLI flag."
        ),
    )
    max_discovery_hits: int = Field(
        default=200,
        ge=10,
        le=2000,
        description="Total hit cap across all discovery patterns per run"
    )


class FalconEyeConfig(BaseModel):
    """Complete FalconEYE configuration."""
    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields
        validate_assignment=True  # Validate on assignment
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    index_registry: IndexRegistryConfig = Field(default_factory=IndexRegistryConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    file_discovery: FileDiscoveryConfig = Field(default_factory=FileDiscoveryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sage: SAGEConfig = Field(default_factory=SAGEConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        import yaml
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)