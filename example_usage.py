#!/usr/bin/env python3
"""Example usage of the Hippocampus Memory MCP Server.

This script demonstrates how to use the memory system for persistent
hippocampus-style memory management across LLM sessions.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta

from memory_mcp_server.storage import MemoryStorage
from memory_mcp_server.tools import MemoryTools


async def example_personal_assistant():
    """Example: Personal assistant memory across multiple conversations."""
    print("=== Personal Assistant Memory Example ===\n")
    
    # Initialize memory system
    storage = MemoryStorage(storage_path="assistant_memory")
    await asyncio.sleep(2)  # Wait for initialization
    
    memory_tools = MemoryTools(storage)
    
    # Session 1: Initial user interaction
    print("Session 1: Learning about the user")
    
    memories_to_store = [
        {
            "text": "User's name is Sarah Chen, she's a data scientist at TechCorp",
            "tags": ["personal", "profession", "identity"],
            "importance_score": 5.0,
            "metadata": {"category": "personal_info", "verified": True}
        },
        {
            "text": "Sarah prefers Python over R for data analysis and uses Jupyter notebooks",
            "tags": ["preference", "programming", "tools"],
            "importance_score": 3.0,
            "metadata": {"category": "work_preference"}
        },
        {
            "text": "Sarah has a meeting with the ML team every Tuesday at 2 PM",
            "tags": ["schedule", "meetings", "work"],
            "importance_score": 4.0,
            "metadata": {"recurring": "weekly", "day": "Tuesday", "time": "14:00"}
        },
        {
            "text": "Sarah is working on a customer churn prediction model using XGBoost",
            "tags": ["project", "ml", "current"],
            "importance_score": 4.5,
            "metadata": {"status": "in_progress", "model": "XGBoost", "domain": "customer_analytics"}
        }
    ]
    
    for memory in memories_to_store:
        result = await memory_tools.memory_write(**memory)
        if result["success"]:
            print(f"✓ Stored: {memory['text'][:50]}...")
        else:
            print(f"✗ Failed: {result['error']}")
    
    print(f"\nSession 1 complete. Stored {len(memories_to_store)} memories.\n")
    
    # Session 2: Later conversation - retrieving context
    print("Session 2: Retrieving user context for a new conversation")
    
    queries = [
        "What is Sarah's professional background?",
        "What programming tools does Sarah prefer?",
        "What project is Sarah currently working on?"
    ]
    
    for query in queries:
        result = await memory_tools.memory_read(
            query_text=query,
            top_k=3,
            min_similarity=0.2
        )
        
        if result["success"]:
            print(f"Query: {query}")
            print(f"Found {result['total_found']} relevant memories:")
            for i, memory in enumerate(result["memories"], 1):
                print(f"  {i}. {memory['text']}")
                print(f"     Similarity score: {memory.get('similarity', 'N/A')}")
                print(f"     Tags: {', '.join(memory['tags'])}")
            print()
        else:
            print(f"Query failed: {result['error']}\n")
    
    return memory_tools


async def example_learning_system():
    """Example: Educational system tracking learning progress."""
    print("=== Learning Progress Tracking Example ===\n")
    
    storage = MemoryStorage(storage_path="learning_memory")
    await asyncio.sleep(2)
    
    memory_tools = MemoryTools(storage)
    
    # Simulate learning sessions over time
    learning_sessions = [
        {
            "text": "User completed Python basics tutorial with 95% score",
            "tags": ["learning", "python", "basics", "completed"],
            "importance_score": 4.0,
            "metadata": {
                "completion_date": "2024-01-10",
                "score": 95,
                "duration_hours": 8,
                "difficulty": "beginner"
            }
        },
        {
            "text": "User struggled with Python decorators concept, needs more practice",
            "tags": ["learning", "python", "decorators", "difficulty"],
            "importance_score": 3.5,
            "metadata": {
                "date": "2024-01-12",
                "attempts": 3,
                "success_rate": 40,
                "needs_review": True
            }
        },
        {
            "text": "User mastered Python list comprehensions with excellent examples",
            "tags": ["learning", "python", "list-comprehensions", "mastered"],
            "importance_score": 3.0,
            "metadata": {
                "date": "2024-01-15",
                "examples_created": 12,
                "confidence": "high"
            }
        },
        {
            "text": "User completed advanced Python OOP tutorial with 88% score",
            "tags": ["learning", "python", "oop", "completed"],
            "importance_score": 4.5,
            "metadata": {
                "completion_date": "2024-01-20",
                "score": 88,
                "duration_hours": 12,
                "difficulty": "advanced"
            }
        }
    ]
    
    print("Storing learning progress...")
    for session in learning_sessions:
        result = await memory_tools.memory_write(**session)
        if result["success"]:
            print(f"✓ Recorded: {session['text'][:60]}...")
    
    # Query learning progress
    print("\nQuerying learning progress...")
    
    progress_query = await memory_tools.memory_read(
        query_text="What Python concepts has the user learned?",
        top_k=10,
        tags=["learning", "python"]
    )
    
    if progress_query["success"]:
        print(f"Found {progress_query['total_found']} learning records:")
        
        completed = [m for m in progress_query["memories"] if "completed" in m["tags"]]
        struggling = [m for m in progress_query["memories"] if "difficulty" in m["tags"]]
        mastered = [m for m in progress_query["memories"] if "mastered" in m["tags"]]
        
        print(f"\nCompleted courses: {len(completed)}")
        for memory in completed:
            metadata = memory.get("metadata", {})
            print(f"  • {memory['text']} (Score: {metadata.get('score', 'N/A')}%)")
        
        print(f"\nChallenging topics: {len(struggling)}")
        for memory in struggling:
            print(f"  • {memory['text']}")
        
        print(f"\nMastered concepts: {len(mastered)}")
        for memory in mastered:
            print(f"  • {memory['text']}")
    
    return memory_tools


async def example_consolidation_and_forgetting():
    """Example: Memory consolidation and forgetting operations."""
    print("\n=== Memory Consolidation and Forgetting Example ===\n")
    
    storage = MemoryStorage(storage_path="consolidation_memory")
    await asyncio.sleep(2)
    
    memory_tools = MemoryTools(storage)
    
    # Store similar memories that should be consolidated
    similar_memories = [
        {
            "text": "User likes coffee in the morning",
            "tags": ["preference", "beverage", "morning"],
            "importance_score": 2.0
        },
        {
            "text": "User prefers coffee over tea for morning drinks",
            "tags": ["preference", "beverage", "morning"],
            "importance_score": 2.5
        },
        {
            "text": "User always has coffee when starting work in the morning",
            "tags": ["preference", "beverage", "morning", "work"],
            "importance_score": 2.2
        },
        {
            "text": "User mentioned they love their morning coffee routine",
            "tags": ["preference", "beverage", "morning", "routine"],
            "importance_score": 2.8
        }
    ]
    
    print("Storing similar memories for consolidation...")
    for memory in similar_memories:
        result = await memory_tools.memory_write(**memory)
        if result["success"]:
            print(f"✓ Stored: {memory['text']}")
    
    # Add some low-importance and temporary memories
    temp_memories = [
        {
            "text": "Debug: temporary test message 1",
            "tags": ["debug", "temporary"],
            "importance_score": 0.5
        },
        {
            "text": "Debug: temporary test message 2",
            "tags": ["debug", "temporary"],
            "importance_score": 0.3
        },
        {
            "text": "Old information that's no longer relevant",
            "tags": ["outdated"],
            "importance_score": 0.8
        }
    ]
    
    for memory in temp_memories:
        await memory_tools.memory_write(**memory)
    
    # Get initial stats
    initial_stats = await memory_tools.memory_stats()
    print(f"\nInitial memory count: {initial_stats['total_memories']}")
    print(f"Initial storage size: {initial_stats['total_size_mb']} MB")
    
    # Consolidate similar memories
    print("\nConsolidating similar memories...")
    consolidation_result = await memory_tools.memory_consolidate(
        similarity_threshold=0.75
    )
    
    if consolidation_result["success"]:
        print(f"✓ Consolidated {consolidation_result['consolidated_groups']} groups")
        print(f"✓ Created {consolidation_result['new_memories_created']} new memories")
        print(f"✓ Removed {consolidation_result['old_memories_removed']} old memories")
        
        if consolidation_result["new_memories"]:
            print("\nNew consolidated memories:")
            for memory in consolidation_result["new_memories"]:
                print(f"  • {memory['text'][:100]}...")
    
    # Forget low-importance memories
    print("\nForgetting low-importance memories...")
    forget_result = await memory_tools.memory_forget(
        min_importance_score=1.0
    )
    
    if forget_result["success"]:
        print(f"✓ Forgot {forget_result['memories_forgotten']} low-importance memories")
        print(f"Memory count: {forget_result['memories_before']} → {forget_result['memories_after']}")
        print(f"Storage size: {forget_result['size_before_mb']} MB → {forget_result['size_after_mb']} MB")
    
    # Forget temporary memories by tag
    print("\nForgetting temporary debug memories...")
    debug_forget_result = await memory_tools.memory_forget(
        tags_to_forget=["debug", "temporary"]
    )
    
    if debug_forget_result["success"]:
        print(f"✓ Forgot {debug_forget_result['memories_forgotten']} debug/temporary memories")
    
    # Final stats
    final_stats = await memory_tools.memory_stats()
    print(f"\nFinal memory count: {final_stats['total_memories']}")
    print(f"Final storage size: {final_stats['total_size_mb']} MB")
    
    return memory_tools


async def demonstrate_cross_session_persistence():
    """Demonstrate memory persistence across server restarts."""
    print("\n=== Cross-Session Persistence Example ===\n")
    
    # Session 1: Store some important information
    print("Session 1: Storing important information...")
    storage1 = MemoryStorage(storage_path="persistent_memory")
    await asyncio.sleep(2)
    memory_tools1 = MemoryTools(storage1)
    
    important_info = [
        {
            "text": "Client's quarterly review is scheduled for March 15th",
            "tags": ["important", "client", "schedule"],
            "importance_score": 5.0,
            "metadata": {"deadline": "2024-03-15", "type": "quarterly_review"}
        },
        {
            "text": "Budget proposal needs final approval from CFO",
            "tags": ["important", "budget", "approval"],
            "importance_score": 4.5,
            "metadata": {"status": "pending_approval", "approver": "CFO"}
        }
    ]
    
    for info in important_info:
        result = await memory_tools1.memory_write(**info)
        if result["success"]:
            print(f"✓ Stored: {info['text']}")
    
    stats1 = await memory_tools1.memory_stats()
    print(f"Session 1: Stored {stats1['total_memories']} memories\n")
    
    # Simulate server restart by creating new instance
    print("--- Simulating server restart ---\n")
    
    # Session 2: Load from persistence
    print("Session 2: Loading from persistent storage...")
    storage2 = MemoryStorage(storage_path="persistent_memory")
    await asyncio.sleep(2)
    memory_tools2 = MemoryTools(storage2)
    
    # Verify data was loaded
    stats2 = await memory_tools2.memory_stats()
    print(f"Session 2: Loaded {stats2['total_memories']} memories from disk")
    
    # Query the persistent data
    query_result = await memory_tools2.memory_read(
        query_text="What important deadlines and approvals are pending?",
        top_k=5,
        tags=["important"]
    )
    
    if query_result["success"]:
        print(f"Found {query_result['total_found']} important items:")
        for memory in query_result["memories"]:
            print(f"  • {memory['text']}")
            metadata = memory.get("metadata", {})
            if "deadline" in metadata:
                print(f"    Deadline: {metadata['deadline']}")
            if "status" in metadata:
                print(f"    Status: {metadata['status']}")
    
    print("\n✓ Cross-session persistence verified!")


async def main():
    """Run all example scenarios."""
    print("Hippocampus Memory MCP Server - Example Usage\n")
    print("=" * 60)
    
    try:
        # Run all examples
        await example_personal_assistant()
        await example_learning_system()
        await example_consolidation_and_forgetting()
        await demonstrate_cross_session_persistence()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. Memories persist across server restarts")
        print("2. Semantic search enables flexible memory retrieval")
        print("3. Consolidation reduces redundancy while preserving information")
        print("4. Forgetting operations maintain system efficiency")
        print("5. Tags and metadata enable sophisticated filtering and organization")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())