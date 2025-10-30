# Hippocampus Memory MCP Server

A Python-based MCP (Model Context Protocol) compliant server that enables Large Language Models to maintain persistent, hippocampus-style memory across sessions. This system allows LLMs to remember user interactions, retrieve and reason over prior knowledge, consolidate repeated information, and forget outdated or irrelevant data.

## Features

- **Persistent Memory Storage**: Vector-based memory storage using FAISS for semantic similarity search
- **MCP Compliant**: Fully compliant with MCP 1.2.0 specification using FastMCP framework
- **Hippocampus-Style Memory**: Implements biological memory patterns with consolidation and forgetting
- **Security First**: Comprehensive input validation, rate limiting, and injection prevention
- **Semantic Search**: Advanced semantic similarity search using sentence transformers
- **Memory Management**: Automatic consolidation of similar memories and intelligent forgetting

## Memory Operations

### Core Tools

1. **`memory_read`**: Retrieve memories based on semantic similarity
2. **`memory_write`**: Store new memories with metadata and tags
3. **`memory_consolidate`**: Merge similar memories to reduce redundancy
4. **`memory_forget`**: Remove memories based on various criteria
5. **`memory_stats`**: Get comprehensive statistics about the memory system

## Installation

### Prerequisites

- Python 3.9 or higher
- MCP SDK 1.2.0 or higher

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd Memory\ MCP
```

2. Install dependencies:
```bash
# Recommended: Use a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install exact versions to avoid PyTorch compatibility issues
pip install -r requirements.txt

# Alternative: If you encounter PyTorch/transformers compatibility issues
pip uninstall torch transformers sentence-transformers -y
pip install torch==2.1.0 transformers==4.35.2 sentence-transformers==2.2.2
```

Or using the project configuration:
```bash
pip install -e .
```

3. Run the server:
```bash
python run_server.py
# or
python -m memory_mcp_server.server
```

## Usage Examples

### Starting the Server

```python
import asyncio
from memory_mcp_server.server import main

# Run the MCP server
asyncio.run(main())
```

### Basic Memory Operations

#### Storing Memories

```python
# Store a simple memory
result = await memory_tools.memory_write(
    text="The user prefers dark mode interface",
    tags=["preference", "ui"],
    importance_score=2.0
)

# Store a memory with metadata
result = await memory_tools.memory_write(
    text="User completed Python tutorial on 2024-01-15",
    metadata={
        "completion_date": "2024-01-15",
        "difficulty": "beginner",
        "score": 95
    },
    tags=["learning", "python", "tutorial"],
    importance_score=3.0
)
```

#### Retrieving Memories

```python
# Search for memories about user preferences
result = await memory_tools.memory_read(
    query_text="What does the user like for interface settings?",
    top_k=5,
    min_similarity=0.3,
    tags=["preference"]
)

# Search with date filtering
result = await memory_tools.memory_read(
    query_text="Python learning progress",
    top_k=10,
    date_range_start="2024-01-01",
    date_range_end="2024-01-31"
)
```

#### Memory Consolidation

```python
# Consolidate similar memories
result = await memory_tools.memory_consolidate(
    similarity_threshold=0.85
)

print(f"Consolidated {result['consolidated_groups']} memory groups")
print(f"Created {result['new_memories_created']} new consolidated memories")
```

#### Forgetting Memories

```python
# Remove old memories (older than 30 days)
result = await memory_tools.memory_forget(
    max_age_days=30
)

# Remove low importance memories
result = await memory_tools.memory_forget(
    min_importance_score=1.0
)

# Remove memories with specific tags
result = await memory_tools.memory_forget(
    tags_to_forget=["temporary", "debug"]
)

# Note: No limit on total memory count - unlimited storage capacity
```

### Complete Example Session

```python
import asyncio
from memory_mcp_server.storage import MemoryStorage
from memory_mcp_server.tools import MemoryTools

async def example_session():
    # Initialize memory system
    storage = MemoryStorage(storage_path="example_memory")
    await asyncio.sleep(2)  # Wait for initialization
    
    memory_tools = MemoryTools(storage)
    
    # Store initial memories
    await memory_tools.memory_write(
        text="User's name is John Smith and he works as a software engineer",
        tags=["personal", "profession"],
        importance_score=5.0
    )
    
    await memory_tools.memory_write(
        text="John prefers Python over JavaScript for backend development",
        tags=["preference", "programming"],
        importance_score=3.0
    )
    
    await memory_tools.memory_write(
        text="John completed a machine learning course in January 2024",
        metadata={"completion_date": "2024-01-15", "certificate": True},
        tags=["learning", "ml", "achievement"],
        importance_score=4.0
    )
    
    # Query memories
    result = await memory_tools.memory_read(
        query_text="Tell me about John's programming preferences",
        top_k=3
    )
    
    print("Found memories:")
    for memory in result["memories"]:
        print(f"- {memory['text']}")
        print(f"  Tags: {memory['tags']}")
        print(f"  Importance: {memory['importance_score']}")
        print()
    
    # Get system statistics
    stats = await memory_tools.memory_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Storage size: {stats['total_size_mb']} MB")

if __name__ == "__main__":
    asyncio.run(example_session())
```

## MCP Integration

### Client Configuration

Add this server to your MCP client configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "memory_mcp_server.server"],
      "env": {}
    }
  }
}
```

### Available MCP Tools

All memory operations are exposed as MCP tools with comprehensive parameter validation:

- `memory_read(query_text, top_k=5, min_similarity=0.1, tags=None, date_range_start=None, date_range_end=None)`
- `memory_write(text, metadata=None, tags=None, importance_score=1.0)`
- `memory_consolidate(similarity_threshold=0.85)`
- `memory_forget(max_age_days=None, min_importance_score=None, tags_to_forget=None)`
- `memory_stats()`

## Architecture

### Core Components

1. **Storage Layer** (`storage.py`): FAISS-based vector storage with sentence transformers
2. **Memory Tools** (`tools.py`): MCP-compliant tool implementations
3. **Security Layer** (`security.py`): Input validation and rate limiting
4. **Server Layer** (`server.py`): FastMCP server implementation
5. **Data Models** (`models.py`): Pydantic models for type safety

### Memory Lifecycle

1. **Write**: Text is embedded using sentence transformers and stored with metadata
2. **Read**: Query text is embedded and used for semantic similarity search
3. **Consolidate**: Similar memories are merged based on similarity threshold
4. **Forget**: Memories are removed based on age, importance, or other criteria

## Security Features

- **Input Validation**: Comprehensive validation of all inputs to prevent injection attacks
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Sanitization**: Automatic sanitization of potentially dangerous content
- **Size Limits**: Enforced limits on text length, metadata size, and tag counts
- **Logging**: Secure logging to stderr to prevent JSON-RPC corruption

## Configuration

### Environment Variables

- `MEMORY_STORAGE_PATH`: Directory for persistent memory storage (default: "memory_data")
- `EMBEDDING_MODEL`: Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `RATE_LIMIT_REQUESTS`: Max requests per window (default: 100)
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: 60)

### Storage Limits

- **Per-item limits** (to prevent resource exhaustion):
  - Maximum text length: 50KB per memory
  - Maximum metadata size: 5KB  
  - Maximum tags per memory: 20
  - Maximum tag length: 50 characters
- **No limit on total memory count** - unlimited storage capacity

## Best Practices

1. **Memory Organization**: Use descriptive tags and appropriate importance scores
2. **Regular Consolidation**: Run consolidation periodically to maintain efficiency
3. **Memory Hygiene**: Use forgetting operations to remove outdated information
4. **Security**: Always validate inputs when integrating with external systems
5. **Monitoring**: Monitor memory statistics to optimize system performance

## Troubleshooting

### Common Issues

1. **Initialization Timeout**: Ensure sufficient time for embedding model loading
2. **Memory Errors**: Check available RAM for large embedding models
3. **Permission Errors**: Verify write permissions for storage directory
4. **Rate Limiting**: Implement proper client-side rate limiting

### Logging

The system logs to stderr by default to maintain MCP compliance. For detailed debugging:

```python
import logging
logging.getLogger('memory_mcp_server').setLevel(logging.DEBUG)
```

## Contributing

1. Follow MCP security guidelines
2. Maintain backward compatibility
3. Add comprehensive tests for new features
4. Update documentation for API changes

## License

This project is provided as an example implementation of MCP-compliant memory management. Please review and adapt the security measures for your specific use case.

## Support

For issues and feature requests, please refer to the MCP documentation at: https://modelcontextprotocol.io/