"""Data models for the memory system."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid


class MemoryEntry(BaseModel):
    """Represents a single memory entry in the hippocampus-style system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The textual content of the memory")
    embedding: List[float] = Field(..., description="Vector embedding of the memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=0, description="Number of times this memory was accessed")
    importance_score: float = Field(default=1.0, description="Importance score for memory consolidation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    
class MemoryQuery(BaseModel):
    """Query model for memory retrieval."""
    
    query_embedding: List[float] = Field(..., description="Vector embedding of the query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of memories to retrieve")
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    date_range: Optional[Dict[str, datetime]] = Field(default=None, description="Date range filter")


class ConsolidationResult(BaseModel):
    """Result of memory consolidation operation."""
    
    consolidated_count: int = Field(..., description="Number of memories consolidated")
    new_memories: List[MemoryEntry] = Field(..., description="Newly created consolidated memories")
    removed_memory_ids: List[str] = Field(..., description="IDs of memories that were removed")


class ForgetCriteria(BaseModel):
    """Criteria for forgetting memories."""
    
    max_age_days: Optional[int] = Field(default=None, description="Maximum age in days")
    min_importance_score: Optional[float] = Field(default=None, description="Minimum importance score")
    tags_to_forget: Optional[List[str]] = Field(default=None, description="Tags of memories to forget")


class MemoryStats(BaseModel):
    """Statistics about the memory system."""
    
    total_memories: int
    total_size_mb: float
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]
    average_importance: float
    tag_distribution: Dict[str, int]