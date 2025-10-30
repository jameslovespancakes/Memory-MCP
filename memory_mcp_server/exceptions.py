"""Custom exceptions for the memory MCP server."""


class MemoryServerError(Exception):
    """Base exception for memory server errors."""
    pass


class ValidationError(MemoryServerError):
    """Raised when input validation fails."""
    pass


class StorageError(MemoryServerError):
    """Raised when storage operations fail."""
    pass


class EmbeddingError(MemoryServerError):
    """Raised when embedding generation fails."""
    pass


class ConsolidationError(MemoryServerError):
    """Raised when memory consolidation fails."""
    pass


class RateLimitError(MemoryServerError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(MemoryServerError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(MemoryServerError):
    """Raised when authorization fails."""
    pass