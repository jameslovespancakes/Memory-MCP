"""Security utilities and validation for the memory MCP server."""

import re
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation utilities for memory operations."""
    
    # Maximum sizes to prevent resource exhaustion per individual item
    MAX_TEXT_LENGTH = 50000  # 50KB per memory
    MAX_METADATA_SIZE = 5000  # 5KB metadata
    MAX_TAGS_COUNT = 20
    MAX_TAG_LENGTH = 50
    MAX_QUERY_LENGTH = 1000
    # No limit on total memory count - only per-item limits
    
    # Dangerous patterns to detect potential injection attempts
    DANGEROUS_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # eval() calls
        r'exec\s*\(',  # exec() calls
        r'import\s+',  # Import statements
        r'__[a-zA-Z_]+__',  # Python dunder methods
        r'\.\./',  # Directory traversal
        r'[<>"\']',  # Basic HTML/SQL injection chars in metadata keys
    ]
    
    @staticmethod
    def validate_text_input(text: str, field_name: str = "text") -> str:
        """Validate and sanitize text input.
        
        Args:
            text: Text to validate
            field_name: Name of the field for error messages
            
        Returns:
            Validated and sanitized text
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError(f"{field_name} must be a string")
        
        if not text.strip():
            raise ValueError(f"{field_name} cannot be empty")
        
        if len(text) > SecurityValidator.MAX_TEXT_LENGTH:
            raise ValueError(f"{field_name} exceeds maximum length of {SecurityValidator.MAX_TEXT_LENGTH} characters")
        
        # Check for dangerous patterns
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected in {field_name}: {pattern}")
                # Remove or escape dangerous content rather than rejecting entirely
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def validate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate and sanitize metadata dictionary.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Validated metadata dictionary
            
        Raises:
            ValueError: If validation fails
        """
        if metadata is None:
            return None
        
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Check total size
        metadata_str = str(metadata)
        if len(metadata_str) > SecurityValidator.MAX_METADATA_SIZE:
            raise ValueError(f"Metadata exceeds maximum size of {SecurityValidator.MAX_METADATA_SIZE} characters")
        
        # Validate keys and values
        validated_metadata = {}
        for key, value in metadata.items():
            # Validate keys
            if not isinstance(key, str):
                raise ValueError("Metadata keys must be strings")
            
            if len(key) > 100:
                raise ValueError("Metadata keys cannot exceed 100 characters")
            
            # Check for dangerous patterns in keys
            clean_key = SecurityValidator.validate_text_input(key, "metadata key")
            
            # Validate values (convert to string and validate)
            if value is not None:
                clean_value = SecurityValidator.validate_text_input(str(value), f"metadata value for key '{clean_key}'")
                validated_metadata[clean_key] = clean_value
            else:
                validated_metadata[clean_key] = None
        
        return validated_metadata
    
    @staticmethod
    def validate_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and sanitize tags list.
        
        Args:
            tags: List of tags to validate
            
        Returns:
            Validated tags list
            
        Raises:
            ValueError: If validation fails
        """
        if tags is None:
            return None
        
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        
        if len(tags) > SecurityValidator.MAX_TAGS_COUNT:
            raise ValueError(f"Cannot have more than {SecurityValidator.MAX_TAGS_COUNT} tags")
        
        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError("All tags must be strings")
            
            if len(tag) > SecurityValidator.MAX_TAG_LENGTH:
                raise ValueError(f"Tag exceeds maximum length of {SecurityValidator.MAX_TAG_LENGTH} characters")
            
            clean_tag = SecurityValidator.validate_text_input(tag, "tag").lower()
            if clean_tag and clean_tag not in validated_tags:
                validated_tags.append(clean_tag)
        
        return validated_tags
    
    @staticmethod
    def validate_importance_score(score: float) -> float:
        """Validate importance score.
        
        Args:
            score: Importance score to validate
            
        Returns:
            Validated importance score
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(score, (int, float)):
            raise ValueError("Importance score must be a number")
        
        if score < 0.0 or score > 10.0:
            raise ValueError("Importance score must be between 0.0 and 10.0")
        
        return float(score)
    
    @staticmethod
    def validate_date_string(date_str: Optional[str]) -> Optional[datetime]:
        """Validate and parse date string.
        
        Args:
            date_str: Date string in ISO format
            
        Returns:
            Parsed datetime object or None
            
        Raises:
            ValueError: If date format is invalid
        """
        if date_str is None:
            return None
        
        if not isinstance(date_str, str):
            raise ValueError("Date must be a string")
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    
    @staticmethod
    def validate_numeric_parameter(value: Any, param_name: str, min_val: float, max_val: float) -> float:
        """Validate numeric parameter within range.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated numeric value
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{param_name} must be a number")
        
        if value < min_val or value > max_val:
            raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
        
        return float(value)


class RateLimiter:
    """Simple rate limiter for memory operations."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str = "default") -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = datetime.now(timezone.utc)
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        cutoff_time = now.timestamp() - self.window_seconds
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False
        
        # Add current request
        self.requests[client_id].append(now.timestamp())
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)