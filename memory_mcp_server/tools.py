"""MCP-compliant memory tools implementation."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
from .storage import MemoryStorage
from .models import MemoryQuery, ForgetCriteria
from .security import SecurityValidator, rate_limiter
from .exceptions import ValidationError, RateLimitError

# Configure logging to stderr
logger = logging.getLogger(__name__)


class MemoryTools:
    """MCP-compliant tools for hippocampus-style memory management."""
    
    def __init__(self, storage: MemoryStorage):
        """Initialize memory tools with storage backend.
        
        Args:
            storage: Memory storage system instance
        """
        self.storage = storage
    
    async def memory_read(
        self, 
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
        tags: Optional[List[str]] = None,
        date_range_start: Optional[str] = None,
        date_range_end: Optional[str] = None,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """Read memories based on semantic similarity to query text.
        
        Args:
            query_text: Text to search for semantically similar memories
            top_k: Maximum number of memories to retrieve (1-100)
            min_similarity: Minimum similarity threshold (0.0-1.0)
            tags: Optional list of tags to filter by
            date_range_start: Optional start date filter (ISO format)
            date_range_end: Optional end date filter (ISO format)
            client_id: Client identifier for rate limiting
            
        Returns:
            Dictionary containing retrieved memories and metadata
        """
        try:
            # Check rate limit
            if not rate_limiter.is_allowed(client_id):
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                    "memories": []
                }
            
            # Validate inputs using security validator
            query_text = SecurityValidator.validate_text_input(query_text, "query_text")
            top_k = int(SecurityValidator.validate_numeric_parameter(top_k, "top_k", 1, 100))
            min_similarity = SecurityValidator.validate_numeric_parameter(min_similarity, "min_similarity", 0.0, 1.0)
            tags = SecurityValidator.validate_tags(tags)
            date_start = SecurityValidator.validate_date_string(date_range_start)
            date_end = SecurityValidator.validate_date_string(date_range_end)
            
            top_k = max(1, min(100, top_k))
            min_similarity = max(0.0, min(1.0, min_similarity))
            
            # Create query embedding
            query_embedding = await self.storage.create_embedding(query_text)
            
            # Parse date range if provided
            date_range = None
            if date_start or date_end:
                date_range = {}
                if date_start:
                    date_range['start'] = date_start
                if date_end:
                    date_range['end'] = date_end
            
            # Create memory query
            memory_query = MemoryQuery(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                tags=tags,
                date_range=date_range
            )
            
            # Retrieve memories
            memories = await self.storage.read_memories(memory_query)
            
            # Format results for MCP response
            memory_results = []
            for memory in memories:
                memory_results.append({
                    "id": memory.id,
                    "text": memory.text,
                    "metadata": memory.metadata,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat(),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "access_count": memory.access_count,
                    "importance_score": memory.importance_score
                })
            
            logger.info(f"Retrieved {len(memory_results)} memories for query: {query_text[:50]}...")
            
            return {
                "success": True,
                "query": query_text,
                "total_found": len(memory_results),
                "memories": memory_results,
                "query_params": {
                    "top_k": top_k,
                    "min_similarity": min_similarity,
                    "tags": tags,
                    "date_range_start": date_range_start,
                    "date_range_end": date_range_end
                }
            }
            
        except Exception as e:
            logger.error(f"Error in memory_read: {e}")
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }
    
    async def memory_write(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance_score: float = 1.0,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """Write a new memory to the storage system.
        
        Args:
            text: Text content of the memory
            metadata: Optional metadata dictionary
            tags: Optional list of tags for categorization
            importance_score: Importance score for consolidation (0.0-10.0)
            client_id: Client identifier for rate limiting
            
        Returns:
            Dictionary containing the stored memory ID and metadata
        """
        try:
            # Check rate limit
            if not rate_limiter.is_allowed(client_id):
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later."
                }
            
            # Validate inputs using security validator
            text = SecurityValidator.validate_text_input(text, "memory text")
            metadata = SecurityValidator.validate_metadata(metadata)
            tags = SecurityValidator.validate_tags(tags)
            importance_score = SecurityValidator.validate_importance_score(importance_score)
            
            # Store the memory
            memory_id = await self.storage.write_memory(
                text=text,
                metadata=metadata,
                tags=tags,
                importance_score=importance_score
            )
            
            logger.info(f"Stored new memory with ID: {memory_id}")
            
            return {
                "success": True,
                "memory_id": memory_id,
                "text": text,
                "metadata": metadata or {},
                "tags": tags or [],
                "importance_score": importance_score,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in memory_write: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def memory_consolidate(
        self,
        similarity_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """Consolidate similar memories to reduce redundancy and improve organization.
        
        Args:
            similarity_threshold: Minimum similarity to consider memories for consolidation (0.0-1.0)
            
        Returns:
            Dictionary containing consolidation results and statistics
        """
        try:
            # Validate inputs
            similarity_threshold = max(0.0, min(1.0, similarity_threshold))
            
            # Perform consolidation
            result = await self.storage.consolidate_memories(similarity_threshold)
            
            # Format results
            new_memories_info = []
            for memory in result.new_memories:
                new_memories_info.append({
                    "id": memory.id,
                    "text": memory.text[:200] + "..." if len(memory.text) > 200 else memory.text,
                    "tags": memory.tags,
                    "importance_score": memory.importance_score,
                    "created_at": memory.created_at.isoformat()
                })
            
            logger.info(f"Consolidated {result.consolidated_count} memory groups")
            
            return {
                "success": True,
                "consolidated_groups": result.consolidated_count,
                "new_memories_created": len(result.new_memories),
                "old_memories_removed": len(result.removed_memory_ids),
                "similarity_threshold": similarity_threshold,
                "new_memories": new_memories_info,
                "removed_memory_ids": result.removed_memory_ids
            }
            
        except Exception as e:
            logger.error(f"Error in memory_consolidate: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def memory_forget(
        self,
        max_age_days: Optional[int] = None,
        min_importance_score: Optional[float] = None,
        tags_to_forget: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Forget memories based on specified criteria to maintain system efficiency.
        
        Args:
            max_age_days: Remove memories older than this many days
            min_importance_score: Remove memories with importance below this threshold
            tags_to_forget: Remove memories with these tags
            
        Returns:
            Dictionary containing forget operation results
        """
        try:
            # Validate inputs
            if max_age_days is not None:
                max_age_days = max(0, max_age_days)
            if min_importance_score is not None:
                min_importance_score = max(0.0, min(10.0, min_importance_score))
            
            # Create forget criteria (no max_memories limit)
            criteria = ForgetCriteria(
                max_age_days=max_age_days,
                min_importance_score=min_importance_score,
                max_memories=None,  # No memory count limit
                tags_to_forget=tags_to_forget
            )
            
            # Get stats before forgetting
            stats_before = await self.storage.get_statistics()
            
            # Perform forget operation
            forgotten_count = await self.storage.forget_memories(criteria)
            
            # Get stats after forgetting
            stats_after = await self.storage.get_statistics()
            
            logger.info(f"Forgot {forgotten_count} memories")
            
            return {
                "success": True,
                "memories_forgotten": forgotten_count,
                "memories_before": stats_before.total_memories,
                "memories_after": stats_after.total_memories,
                "size_before_mb": round(stats_before.total_size_mb, 2),
                "size_after_mb": round(stats_after.total_size_mb, 2),
                "criteria_applied": {
                    "max_age_days": max_age_days,
                    "min_importance_score": min_importance_score,
                    "tags_to_forget": tags_to_forget
                }
            }
            
        except Exception as e:
            logger.error(f"Error in memory_forget: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system.
        
        Returns:
            Dictionary containing memory system statistics
        """
        try:
            stats = await self.storage.get_statistics()
            
            return {
                "success": True,
                "total_memories": stats.total_memories,
                "total_size_mb": round(stats.total_size_mb, 2),
                "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
                "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
                "average_importance": round(stats.average_importance, 2),
                "tag_distribution": stats.tag_distribution,
                "most_common_tags": sorted(
                    stats.tag_distribution.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
            
        except Exception as e:
            logger.error(f"Error in memory_stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }