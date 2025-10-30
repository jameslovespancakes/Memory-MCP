<div align="center">

# 🧠 Hippocampus Memory MCP Server

### *Persistent, Semantic Memory for Large Language Models*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.2.0-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture)

---

</div>

## 📖 Overview

A Python-based **Model Context Protocol (MCP)** server that gives LLMs persistent, hippocampus-inspired memory across sessions. Store, retrieve, consolidate, and forget memories using semantic similarity search powered by vector embeddings.

**Why Hippocampus?** Just like the human brain's hippocampus consolidates short-term memories into long-term storage, this server intelligently manages LLM memory through biological patterns:
- 🔄 **Consolidation** - Merge similar memories to reduce redundancy
- 🧹 **Forgetting** - Remove outdated information based on age/importance
- 🔍 **Semantic Retrieval** - Find relevant memories through meaning, not keywords

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗄️ **Vector Storage** | FAISS-powered semantic similarity search |
| 🎯 **MCP Compliant** | Full MCP 1.2.0 spec compliance via FastMCP |
| 🧬 **Bio-Inspired** | Hippocampus-style consolidation and forgetting |
| 🔒 **Security** | Input validation, rate limiting, injection prevention |
| 🔎 **Semantic Search** | Sentence transformer embeddings (CPU-optimized) |
| ♾️ **Unlimited Storage** | No memory count limits, only per-item size limits |
| 🆓 **100% Free** | Local embedding model - no API costs |

## 🚀 Quick Start

### 5 Core MCP Tools

```python
memory_read         # 🔍 Retrieve memories by semantic similarity
memory_write        # ✍️  Store new memories with tags & metadata
memory_consolidate  # 🔄 Merge similar memories
memory_forget       # 🧹 Remove memories by age/importance/tags
memory_stats        # 📊 Get system statistics
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- ~200MB disk space (for embedding model)

### Setup in 3 Steps

```bash
# 1. Clone the repository
git clone https://github.com/jameslovespancakes/Memory-MCP.git
cd Memory-MCP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python -m memory_mcp_server.server
```

### Claude Desktop Integration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "memory_mcp_server.server"],
      "cwd": "/path/to/Memory-MCP"
    }
  }
}
```

> **🎉 That's it!** Claude will now have persistent memory across conversations.

## 📚 Documentation

### Memory Operations via MCP

Once connected to Claude, use natural language:

```
"Remember that I prefer Python for backend development"
→ Claude calls memory_write()

"What do you know about my programming preferences?"
→ Claude calls memory_read()

"Consolidate similar memories to clean up storage"
→ Claude calls memory_consolidate()
```

### Direct API Usage

#### ✍️ Writing Memories

```python
from memory_mcp_server.storage import MemoryStorage
from memory_mcp_server.tools import MemoryTools

storage = MemoryStorage(storage_path="my_memory")
await storage._ensure_initialized()
tools = MemoryTools(storage)

# Store with tags and importance
await tools.memory_write(
    text="User prefers dark mode UI",
    tags=["preference", "ui"],
    importance_score=3.0,
    metadata={"category": "settings"}
)
```

#### 🔍 Reading Memories

```python
# Semantic search
result = await tools.memory_read(
    query_text="What are my UI preferences?",
    top_k=5,
    min_similarity=0.3
)

# Filter by tags and date
result = await tools.memory_read(
    query_text="Python learning",
    tags=["learning", "python"],
    date_range_start="2024-01-01"
)
```

#### 🔄 Consolidating Memories

```python
# Merge similar memories (threshold: 0.85)
result = await tools.memory_consolidate(similarity_threshold=0.85)
print(f"Merged {result['consolidated_groups']} groups")
```

#### 🧹 Forgetting Memories

```python
# Remove by age
await tools.memory_forget(max_age_days=30)

# Remove by importance
await tools.memory_forget(min_importance_score=2.0)

# Remove by tags
await tools.memory_forget(tags_to_forget=["temporary"])
```

### Testing

Run the included test suite:

```bash
python test_memory.py
```

This tests all 5 operations with sample data.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  MCP Client (Claude Desktop, etc.)                  │
└───────────────────┬─────────────────────────────────┘
                    │ JSON-RPC over stdio
┌───────────────────▼─────────────────────────────────┐
│  FastMCP Server (server.py)                         │
│  ├─ memory_read                                     │
│  ├─ memory_write                                    │
│  ├─ memory_consolidate                              │
│  ├─ memory_forget                                   │
│  └─ memory_stats                                    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Memory Tools (tools.py)                            │
│  ├─ Input validation & sanitization                │
│  └─ Rate limiting (100 req/min)                    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Storage Layer (storage.py)                         │
│  ├─ Sentence Transformers (all-MiniLM-L6-v2)       │
│  ├─ FAISS Vector Index (cosine similarity)         │
│  └─ JSON persistence (memories.json)               │
└─────────────────────────────────────────────────────┘
```

### 🔄 Memory Lifecycle

| Step | Process | Technology |
|------|---------|------------|
| 📝 **Write** | Text → 384-dim vector embedding | Sentence Transformers (CPU) |
| 💾 **Store** | Normalized vector → FAISS index | FAISS IndexFlatIP |
| 🔍 **Search** | Query → embedding → top-k similar | Cosine similarity |
| 🔄 **Consolidate** | Group similar (>0.85) → merge | Vector clustering |
| 🧹 **Forget** | Filter by age/importance/tags → delete | Metadata filtering |

## 🔒 Security

| Protection | Implementation |
|------------|----------------|
| 🛡️ **Injection Prevention** | Regex filtering of script tags, eval(), path traversal |
| ⏱️ **Rate Limiting** | 100 requests per 60-second window per client |
| 📏 **Size Limits** | 50KB text, 5KB metadata, 20 tags per memory |
| ✅ **Input Validation** | Pydantic models + custom sanitization |
| 🔐 **Safe Logging** | stderr only (prevents JSON-RPC corruption) |

## ⚙️ Configuration

### Environment Variables

```bash
MEMORY_STORAGE_PATH="memory_data"           # Storage directory
EMBEDDING_MODEL="all-MiniLM-L6-v2"          # Model name
RATE_LIMIT_REQUESTS=100                     # Max requests
RATE_LIMIT_WINDOW=60                        # Time window (seconds)
```

### Storage Limits

- ✅ **Unlimited total memories** (no count limit)
- ⚠️ Per-memory limits: 50KB text, 5KB metadata, 20 tags

## 🐛 Troubleshooting

<details>
<summary><b>Model won't download</b></summary>

First run downloads `all-MiniLM-L6-v2` (~90MB). Ensure internet connection and `~/.cache/` write permissions.
</details>

<details>
<summary><b>PyTorch compatibility errors</b></summary>

```bash
pip uninstall torch transformers sentence-transformers -y
pip install torch==2.1.0 transformers==4.35.2 sentence-transformers==2.2.2
```
</details>

<details>
<summary><b>Memory errors on large operations</b></summary>

The model runs on CPU. Ensure 2GB+ free RAM. Reduce `top_k` in read operations if needed.
</details>

## 📝 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

PRs welcome! Please:
- Follow MCP security guidelines
- Add tests for new features
- Update documentation

## 🔗 Resources

- [Model Context Protocol Docs](https://modelcontextprotocol.io/)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

---

<div align="center">

**Built with 🧠 for persistent LLM memory**

[Report Bug](https://github.com/jameslovespancakes/Memory-MCP/issues) · [Request Feature](https://github.com/jameslovespancakes/Memory-MCP/issues)

</div>