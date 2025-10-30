"""FastMCP server implementation for hippocampus-style memory management."""

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from .storage import MemoryStorage
from .tools import MemoryTools

# Configure logging to stderr to avoid JSON-RPC corruption
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Hippocampus Memory Server")

# Global storage and tools instances
storage: Optional[MemoryStorage] = None
memory_tools: Optional[MemoryTools] = None


@mcp.tool()
async def memory_read(
    query_text: str,
    top_k: int = 5,
    min_similarity: float = 0.1,
    tags: Optional[List[str]] = None,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None
) -> List[TextContent]:
    """Retrieve memories based on semantic similarity to query text.
    
    Searches the memory system for entries that are semantically similar to the query text.
    Returns the most relevant memories based on vector similarity, with support for filtering
    by tags and date ranges.
    
    Args:
        query_text: Text to search for semantically similar memories
        top_k: Maximum number of memories to retrieve (1-100, default: 5)
        min_similarity: Minimum similarity threshold (0.0-1.0, default: 0.1)
        tags: Optional list of tags to filter by
        date_range_start: Optional start date filter (ISO format: YYYY-MM-DD)
        date_range_end: Optional end date filter (ISO format: YYYY-MM-DD)
        
    Returns:
        List of TextContent containing retrieved memories and metadata
    """
    if not memory_tools:
        return [TextContent(
            type="text",
            text="Error: Memory system not initialized"
        )]
    
    try:
        result = await memory_tools.memory_read(
            query_text=query_text,
            top_k=top_k,
            min_similarity=min_similarity,
            tags=tags,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            client_id="mcp_client"
        )
        
        if not result["success"]:
            return [TextContent(
                type="text",
                text=f"Memory read failed: {result.get('error', 'Unknown error')}"
            )]
        
        # Format response
        response_text = f"Found {result['total_found']} memories for query: '{query_text}'\n\n"
        
        for i, memory in enumerate(result["memories"], 1):
            response_text += f"--- Memory {i} ---\n"
            response_text += f"ID: {memory['id']}\n"
            response_text += f"Text: {memory['text']}\n"
            response_text += f"Tags: {', '.join(memory['tags']) if memory['tags'] else 'None'}\n"
            response_text += f"Created: {memory['created_at']}\n"
            response_text += f"Access Count: {memory['access_count']}\n"
            response_text += f"Importance: {memory['importance_score']}\n"
            if memory['metadata']:
                response_text += f"Metadata: {memory['metadata']}\n"
            response_text += "\n"
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Error in memory_read tool: {e}")
        return [TextContent(
            type="text",
            text=f"Internal error during memory read: {str(e)}"
        )]


@mcp.tool()
async def memory_write(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    importance_score: float = 1.0
) -> List[TextContent]:
    """Store a new memory in the system.
    
    Creates a new memory entry with the provided text content, optional metadata,
    tags for categorization, and importance score for future consolidation decisions.
    
    Args:
        text: Text content of the memory (required)
        metadata: Optional dictionary of additional metadata
        tags: Optional list of tags for categorization
        importance_score: Importance score for consolidation (0.0-10.0, default: 1.0)
        
    Returns:
        List of TextContent containing the stored memory details
    """
    if not memory_tools:
        return [TextContent(
            type="text",
            text="Error: Memory system not initialized"
        )]
    
    try:
        result = await memory_tools.memory_write(
            text=text,
            metadata=metadata,
            tags=tags,
            importance_score=importance_score,
            client_id="mcp_client"
        )
        
        if not result["success"]:
            return [TextContent(
                type="text",
                text=f"Memory write failed: {result.get('error', 'Unknown error')}"
            )]
        
        response_text = f"Successfully stored new memory!\n\n"
        response_text += f"Memory ID: {result['memory_id']}\n"
        response_text += f"Text: {result['text']}\n"
        response_text += f"Tags: {', '.join(result['tags']) if result['tags'] else 'None'}\n"
        response_text += f"Importance Score: {result['importance_score']}\n"
        response_text += f"Created: {result['created_at']}\n"
        
        if result['metadata']:
            response_text += f"Metadata: {result['metadata']}\n"
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Error in memory_write tool: {e}")
        return [TextContent(
            type="text",
            text=f"Internal error during memory write: {str(e)}"
        )]


@mcp.tool()
async def memory_consolidate(
    similarity_threshold: float = 0.85
) -> List[TextContent]:
    """Consolidate similar memories to reduce redundancy.
    
    Analyzes existing memories for similarity and merges highly similar entries
    into consolidated summaries. This helps maintain system efficiency and
    reduces redundant information while preserving important details.
    
    Args:
        similarity_threshold: Minimum similarity to consolidate (0.0-1.0, default: 0.85)
        
    Returns:
        List of TextContent containing consolidation results and statistics
    """
    if not memory_tools:
        return [TextContent(
            type="text",
            text="Error: Memory system not initialized"
        )]
    
    try:
        result = await memory_tools.memory_consolidate(
            similarity_threshold=similarity_threshold
        )
        
        if not result["success"]:
            return [TextContent(
                type="text",
                text=f"Memory consolidation failed: {result.get('error', 'Unknown error')}"
            )]
        
        response_text = f"Memory consolidation completed!\n\n"
        response_text += f"Groups consolidated: {result['consolidated_groups']}\n"
        response_text += f"New memories created: {result['new_memories_created']}\n"
        response_text += f"Old memories removed: {result['old_memories_removed']}\n"
        response_text += f"Similarity threshold: {result['similarity_threshold']}\n\n"
        
        if result['new_memories']:
            response_text += "New consolidated memories:\n"
            for i, memory in enumerate(result['new_memories'], 1):
                response_text += f"{i}. ID: {memory['id']}\n"
                response_text += f"   Text: {memory['text']}\n"
                response_text += f"   Tags: {', '.join(memory['tags']) if memory['tags'] else 'None'}\n"
                response_text += f"   Importance: {memory['importance_score']}\n\n"
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Error in memory_consolidate tool: {e}")
        return [TextContent(
            type="text",
            text=f"Internal error during memory consolidation: {str(e)}"
        )]


@mcp.tool()
async def memory_forget(
    max_age_days: Optional[int] = None,
    min_importance_score: Optional[float] = None,
    tags_to_forget: Optional[List[str]] = None
) -> List[TextContent]:
    """Forget memories based on specified criteria.
    
    Removes memories from the system based on age, importance, or specific tags. 
    This helps maintain system efficiency by removing outdated or less relevant 
    information. Note: There is no limit on total memory count.
    
    Args:
        max_age_days: Remove memories older than this many days
        min_importance_score: Remove memories with importance below this threshold
        tags_to_forget: Remove memories with these specific tags
        
    Returns:
        List of TextContent containing forget operation results and statistics
    """
    if not memory_tools:
        return [TextContent(
            type="text",
            text="Error: Memory system not initialized"
        )]
    
    try:
        result = await memory_tools.memory_forget(
            max_age_days=max_age_days,
            min_importance_score=min_importance_score,
            tags_to_forget=tags_to_forget
        )
        
        if not result["success"]:
            return [TextContent(
                type="text",
                text=f"Memory forget operation failed: {result.get('error', 'Unknown error')}"
            )]
        
        response_text = f"Memory forget operation completed!\n\n"
        response_text += f"Memories forgotten: {result['memories_forgotten']}\n"
        response_text += f"Memories before: {result['memories_before']}\n"
        response_text += f"Memories after: {result['memories_after']}\n"
        response_text += f"Size before: {result['size_before_mb']} MB\n"
        response_text += f"Size after: {result['size_after_mb']} MB\n\n"
        
        response_text += "Criteria applied:\n"
        criteria = result['criteria_applied']
        if criteria['max_age_days']:
            response_text += f"- Maximum age: {criteria['max_age_days']} days\n"
        if criteria['min_importance_score']:
            response_text += f"- Minimum importance: {criteria['min_importance_score']}\n"
        if criteria['tags_to_forget']:
            response_text += f"- Tags to forget: {', '.join(criteria['tags_to_forget'])}\n"
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Error in memory_forget tool: {e}")
        return [TextContent(
            type="text",
            text=f"Internal error during memory forget operation: {str(e)}"
        )]


@mcp.tool()
async def memory_stats() -> List[TextContent]:
    """Get statistics about the memory system.
    
    Provides comprehensive statistics about the current state of the memory system,
    including total memories, storage size, age information, importance scores,
    and tag distribution.
    
    Returns:
        List of TextContent containing detailed memory system statistics
    """
    if not memory_tools:
        return [TextContent(
            type="text",
            text="Error: Memory system not initialized"
        )]
    
    try:
        result = await memory_tools.memory_stats()
        
        if not result["success"]:
            return [TextContent(
                type="text",
                text=f"Memory stats retrieval failed: {result.get('error', 'Unknown error')}"
            )]
        
        response_text = f"Memory System Statistics\n{'=' * 30}\n\n"
        response_text += f"Total memories: {result['total_memories']}\n"
        response_text += f"Total size: {result['total_size_mb']} MB\n"
        response_text += f"Average importance: {result['average_importance']}\n"
        
        if result['oldest_memory']:
            response_text += f"Oldest memory: {result['oldest_memory']}\n"
        if result['newest_memory']:
            response_text += f"Newest memory: {result['newest_memory']}\n"
        
        response_text += f"\nTag distribution:\n"
        if result['tag_distribution']:
            for tag, count in result['most_common_tags']:
                response_text += f"  {tag}: {count}\n"
        else:
            response_text += "  No tags found\n"
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Error in memory_stats tool: {e}")
        return [TextContent(
            type="text",
            text=f"Internal error during stats retrieval: {str(e)}"
        )]


async def initialize_memory_system():
    """Initialize the memory storage system and tools."""
    global storage, memory_tools
    
    try:
        logger.info("Initializing memory storage system...")
        storage = MemoryStorage(storage_path="memory_data")
        
        # Initialize storage system
        await storage._ensure_initialized()
        
        memory_tools = MemoryTools(storage)
        logger.info("Memory system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        raise


async def main():
    """Main entry point for the MCP server."""
    try:
        # Initialize memory system
        await initialize_memory_system()
        
        # Configure server info
        mcp.server_info = {
            "name": "Hippocampus Memory Server",
            "version": "1.0.0",
            "description": "MCP-compliant server for persistent hippocampus-style memory management"
        }
        
        logger.info("Starting Hippocampus Memory MCP Server...")
        
        # Run the server with proper event loop handling
        async with AsyncExitStack():
            await mcp.run_stdio_async()
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())