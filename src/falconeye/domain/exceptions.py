"""Domain exceptions for FalconEYE."""


class FalconEyeDomainError(Exception):
    """Base exception for domain errors."""
    pass


class InvalidCodebaseError(FalconEyeDomainError):
    """Raised when codebase is invalid or inaccessible."""
    pass


class UnsupportedLanguageError(FalconEyeDomainError):
    """Raised when language is not supported."""
    pass


class InvalidChunkError(FalconEyeDomainError):
    """Raised when code chunk is invalid."""
    pass


class AnalysisError(FalconEyeDomainError):
    """Raised when AI analysis fails."""
    pass


class InvalidSecurityFindingError(FalconEyeDomainError):
    """Raised when security finding is malformed."""
    pass


class LanguageDetectionError(FalconEyeDomainError):
    """Raised when automatic language detection fails."""
    pass


class OllamaServiceError(FalconEyeDomainError):
    """Base exception for Ollama service errors."""
    pass


class OllamaConnectionError(OllamaServiceError):
    """Raised when cannot connect to Ollama service."""
    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when Ollama model is not found."""
    pass


class OllamaTimeoutError(OllamaServiceError):
    """Raised when Ollama request times out."""
    pass