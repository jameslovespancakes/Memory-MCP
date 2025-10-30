"""Memory storage implementation using FAISS vector database."""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import faiss
from .models import MemoryEntry, MemoryQuery, ConsolidationResult, ForgetCriteria, MemoryStats

# Import sentence_transformers at runtime to handle compatibility issues
SentenceTransformer = None

# Configure logging to stderr to avoid JSON-RPC corruption
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class MemoryStorage:
    """Hippocampus-style memory storage using FAISS for vector similarity."""
    
    def __init__(self, storage_path: str = "memory_data", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the memory storage system.
        
        Args:
            storage_path: Directory to store persistent memory data
            embedding_model: Name of the sentence transformer model to use
        """
        self.storage_path = storage_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # FAISS index for vector similarity search
        self.index = None
        
        # In-memory storage for memory entries (indexed by FAISS position)
        self.memories: Dict[int, MemoryEntry] = {}
        self.id_to_position: Dict[str, int] = {}
        self.next_position = 0
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize components - defer to avoid event loop conflicts
        self._initialized = False
    
    async def _initialize(self):
        """Initialize the embedding model and load existing data."""
        try:
            # Import sentence_transformers at runtime with error handling
            global SentenceTransformer
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as ST
                    SentenceTransformer = ST
                    logger.info("Successfully imported sentence_transformers")
                except Exception as import_error:
                    logger.error(f"Failed to import sentence_transformers: {import_error}")
                    logger.info("Trying alternative PyTorch compatibility fix...")
                    
                    # Apply PyTorch compatibility patch
                    try:
                        import torch.utils._pytree as pytree
                        if hasattr(pytree, '_register_pytree_node') and not hasattr(pytree, 'register_pytree_node'):
                            pytree.register_pytree_node = pytree._register_pytree_node
                            logger.info("Applied PyTorch pytree compatibility patch")
                        
                        from sentence_transformers import SentenceTransformer as ST
                        SentenceTransformer = ST
                        logger.info("Successfully imported sentence_transformers after patch")
                    except Exception as patch_error:
                        logger.error(f"Compatibility patch failed: {patch_error}")
                        raise ImportError(f"Cannot import sentence_transformers. Original error: {import_error}. Patch error: {patch_error}")
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cpu')
            logger.info("Model forced to run on CPU")
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Load existing memories
            await self._load_memories()
            
            logger.info(f"Memory storage initialized with {len(self.memories)} existing memories")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory storage: {e}")
            logger.error("Please check your PyTorch and transformers installation.")
            logger.error("Try: pip install torch==1.13.1 transformers==4.35.2 sentence-transformers==2.2.2")
            raise
    
    async def _ensure_initialized(self):
        """Ensure the storage system is initialized."""
        if not self._initialized:
            await self._initialize()
    
    async def _load_memories(self):
        """Load memories from persistent storage."""
        try:
            # Load FAISS index
            index_path = os.path.join(self.storage_path, "memory_index.faiss")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            
            # Load memory entries
            memories_path = os.path.join(self.storage_path, "memories.json")
            if os.path.exists(memories_path):
                with open(memories_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for mem_data in data['memories']:
                    memory = MemoryEntry(**mem_data)
                    position = len(self.memories)
                    self.memories[position] = memory
                    self.id_to_position[memory.id] = position
                
                self.next_position = len(self.memories)
                logger.info(f"Loaded {len(self.memories)} memories from storage")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    async def _save_memories(self):
        """Save memories to persistent storage."""
        try:
            # Save FAISS index
            index_path = os.path.join(self.storage_path, "memory_index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save memory entries
            memories_path = os.path.join(self.storage_path, "memories.json")
            memories_data = {
                'memories': [memory.model_dump() for memory in self.memories.values()],
                'next_position': self.next_position
            }
            
            with open(memories_path, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2, default=str)
                
            logger.info(f"Saved {len(self.memories)} memories to storage")
            
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            raise
    
    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        emb_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for given text."""
        await self._ensure_initialized()
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def write_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                          tags: Optional[List[str]] = None, importance_score: float = 1.0) -> str:
        """Write a new memory to storage.
        
        Args:
            text: The textual content of the memory
            metadata: Additional metadata
            tags: Tags for categorization
            importance_score: Importance score for consolidation
            
        Returns:
            Memory ID of the stored memory
        """
        try:
            # Create embedding
            embedding = await self.create_embedding(text)
            
            # Create memory entry
            memory = MemoryEntry(
                text=text,
                embedding=embedding,
                metadata=metadata or {},
                tags=tags or [],
                importance_score=importance_score
            )
            
            # Add to FAISS index
            normalized_emb = self._normalize_embedding(embedding)
            self.index.add(normalized_emb)
            
            # Store memory
            position = self.next_position
            self.memories[position] = memory
            self.id_to_position[memory.id] = position
            self.next_position += 1
            
            # Save to disk
            await self._save_memories()
            
            logger.info(f"Stored new memory with ID: {memory.id}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Error writing memory: {e}")
            raise
    
    async def read_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Read memories based on query.
        
        Args:
            query: Memory query parameters
            
        Returns:
            List of matching memory entries
        """
        try:
            if len(self.memories) == 0:
                return []
            
            # Search in FAISS index
            normalized_query = self._normalize_embedding(query.query_embedding)
            scores, indices = self.index.search(normalized_query, min(query.top_k, len(self.memories)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < query.min_similarity:
                    continue
                
                memory = self.memories[idx]
                
                # Apply filters
                if query.tags and not any(tag in memory.tags for tag in query.tags):
                    continue
                    
                if query.date_range:
                    if 'start' in query.date_range and memory.created_at < query.date_range['start']:
                        continue
                    if 'end' in query.date_range and memory.created_at > query.date_range['end']:
                        continue
                
                # Update access statistics
                memory.last_accessed = datetime.now(timezone.utc)
                memory.access_count += 1
                
                results.append(memory)
            
            # Save updated access statistics
            if results:
                await self._save_memories()
                
            logger.info(f"Retrieved {len(results)} memories for query")
            return results
            
        except Exception as e:
            logger.error(f"Error reading memories: {e}")
            raise
    
    async def consolidate_memories(self, similarity_threshold: float = 0.85) -> ConsolidationResult:
        """Consolidate similar memories into summary chunks.
        
        Args:
            similarity_threshold: Minimum similarity to consider memories for consolidation
            
        Returns:
            Consolidation result with statistics
        """
        try:
            if len(self.memories) < 2:
                return ConsolidationResult(
                    consolidated_count=0,
                    new_memories=[],
                    removed_memory_ids=[]
                )
            
            # Find similar memory groups
            memory_list = list(self.memories.values())
            groups = []
            processed = set()
            
            for i, memory1 in enumerate(memory_list):
                if memory1.id in processed:
                    continue
                
                group = [memory1]
                processed.add(memory1.id)
                
                for j, memory2 in enumerate(memory_list[i+1:], i+1):
                    if memory2.id in processed:
                        continue
                    
                    # Calculate similarity
                    emb1 = np.array(memory1.embedding)
                    emb2 = np.array(memory2.embedding)
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    if similarity >= similarity_threshold:
                        group.append(memory2)
                        processed.add(memory2.id)
                
                if len(group) > 1:
                    groups.append(group)
            
            # Consolidate groups
            new_memories = []
            removed_ids = []
            
            for group in groups:
                # Create consolidated memory
                consolidated_text = " ".join([mem.text for mem in group])
                combined_tags = list(set([tag for mem in group for tag in mem.tags]))
                avg_importance = sum(mem.importance_score for mem in group) / len(group)
                
                # Combine metadata
                combined_metadata = {}
                for mem in group:
                    combined_metadata.update(mem.metadata)
                combined_metadata['consolidated_from'] = [mem.id for mem in group]
                combined_metadata['consolidation_date'] = datetime.now(timezone.utc).isoformat()
                
                # Create new consolidated memory
                new_id = await self.write_memory(
                    text=consolidated_text,
                    metadata=combined_metadata,
                    tags=combined_tags,
                    importance_score=avg_importance
                )
                
                new_memory = None
                for mem in self.memories.values():
                    if mem.id == new_id:
                        new_memory = mem
                        break
                
                if new_memory:
                    new_memories.append(new_memory)
                
                # Remove original memories
                for mem in group:
                    await self._remove_memory(mem.id)
                    removed_ids.append(mem.id)
            
            result = ConsolidationResult(
                consolidated_count=len(groups),
                new_memories=new_memories,
                removed_memory_ids=removed_ids
            )
            
            logger.info(f"Consolidated {result.consolidated_count} memory groups")
            return result
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            raise
    
    async def forget_memories(self, criteria: ForgetCriteria) -> int:
        """Forget memories based on criteria.
        
        Args:
            criteria: Criteria for selecting memories to forget
            
        Returns:
            Number of memories forgotten
        """
        try:
            memories_to_remove = []
            current_time = datetime.now(timezone.utc)
            
            for memory in self.memories.values():
                should_forget = False
                
                # Age criteria
                if criteria.max_age_days:
                    age = (current_time - memory.created_at).days
                    if age > criteria.max_age_days:
                        should_forget = True
                
                # Importance criteria
                if criteria.min_importance_score:
                    if memory.importance_score < criteria.min_importance_score:
                        should_forget = True
                
                # Tag criteria
                if criteria.tags_to_forget:
                    if any(tag in memory.tags for tag in criteria.tags_to_forget):
                        should_forget = True
                
                if should_forget:
                    memories_to_remove.append(memory.id)
            
            # No max_memories constraint - unlimited memory storage
            # Only remove based on age, importance, and tags
            
            # Remove selected memories
            for memory_id in memories_to_remove:
                await self._remove_memory(memory_id)
            
            logger.info(f"Forgot {len(memories_to_remove)} memories")
            return len(memories_to_remove)
            
        except Exception as e:
            logger.error(f"Error forgetting memories: {e}")
            raise
    
    async def _remove_memory(self, memory_id: str):
        """Remove a memory from storage."""
        if memory_id not in self.id_to_position:
            return
        
        position = self.id_to_position[memory_id]
        
        # Remove from FAISS index (rebuild entire index)
        if len(self.memories) > 1:
            remaining_embeddings = []
            new_memories = {}
            new_id_to_position = {}
            new_position = 0
            
            for pos, memory in self.memories.items():
                if memory.id != memory_id:
                    remaining_embeddings.append(memory.embedding)
                    new_memories[new_position] = memory
                    new_id_to_position[memory.id] = new_position
                    new_position += 1
            
            # Rebuild FAISS index
            if remaining_embeddings:
                self.index = faiss.IndexFlatIP(self.dimension)
                embeddings_array = np.array(remaining_embeddings, dtype=np.float32)
                # Normalize embeddings
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_array = embeddings_array / np.where(norms == 0, 1, norms)
                self.index.add(embeddings_array)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            
            self.memories = new_memories
            self.id_to_position = new_id_to_position
            self.next_position = new_position
        else:
            # Last memory, reset everything
            self.index = faiss.IndexFlatIP(self.dimension)
            self.memories = {}
            self.id_to_position = {}
            self.next_position = 0
        
        await self._save_memories()
    
    async def get_statistics(self) -> MemoryStats:
        """Get memory system statistics."""
        try:
            if not self.memories:
                return MemoryStats(
                    total_memories=0,
                    total_size_mb=0.0,
                    oldest_memory=None,
                    newest_memory=None,
                    average_importance=0.0,
                    tag_distribution={}
                )
            
            memories = list(self.memories.values())
            total_size = sum(len(json.dumps(mem.model_dump()).encode('utf-8')) for mem in memories)
            
            # Calculate tag distribution
            tag_dist = {}
            for memory in memories:
                for tag in memory.tags:
                    tag_dist[tag] = tag_dist.get(tag, 0) + 1
            
            return MemoryStats(
                total_memories=len(memories),
                total_size_mb=total_size / (1024 * 1024),
                oldest_memory=min(mem.created_at for mem in memories),
                newest_memory=max(mem.created_at for mem in memories),
                average_importance=sum(mem.importance_score for mem in memories) / len(memories),
                tag_distribution=tag_dist
            )
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise