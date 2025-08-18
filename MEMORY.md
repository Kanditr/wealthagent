# Memory System Documentation

## Overview

The Wealth Agent uses a **hybrid memory system** combining:
- **SQLite checkpointing** for persistent conversation state across application restarts
- **LangMem SDK** with **InMemoryStore** for fast semantic memory operations during active sessions

This provides both persistence and performance.

## Memory Features

### Automatic Memory Management
- **Stores information automatically**: The agent uses `create_manage_memory_tool` to extract and save important user information
- **Retrieves context intelligently**: Uses `create_search_memory_tool` to find relevant past conversations
- **User-specific namespacing**: Each user gets isolated memory space for privacy

### What Gets Remembered
- **Personal Information**: Name, age, contact details
- **Investment Preferences**: Risk tolerance, preferred asset classes, investment style
- **Financial Goals**: Retirement targets, major purchases, timelines
- **Past Conversations**: Previous advice given and questions asked
- **User Feedback**: Preferences about communication style and advice format

## Technical Implementation

### Storage Backend
- **Current**: `InMemoryStore` with OpenAI embeddings (text-embedding-3-small)
- **Embedding Dimensions**: 1536-dimensional vectors for semantic search
- **Performance**: NumPy-optimized vector operations

### Memory Tools
- `create_manage_memory_tool(namespace=("memories", user_id))`: Stores new information
- `create_search_memory_tool(namespace=("memories", user_id))`: Retrieves relevant memories

## Memory Persistence

### Hybrid Persistence Model
- ✅ **SQLite Checkpointing**: Conversation state persists across application restarts
- ✅ **Automatic Database Creation**: `wealth_agent_memories.db` created in project directory
- ✅ **Fast Session Memory**: InMemoryStore provides quick access during active conversations
- ✅ **Seamless Integration**: No user configuration required

### Privacy & Isolation
- ✅ **User Namespacing**: Each user ID gets completely isolated memory
- ✅ **No Cross-User Data**: Users cannot access other users' stored information
- ✅ **Secure by Design**: Memory tools prevent data leakage between users

## Production Considerations

### For Production Use
To enable true persistent memory across server restarts:

1. **Database Storage**: Replace InMemoryStore with AsyncPostgresStore
2. **Configuration Example**:
   ```python
   from langgraph.store.postgres import AsyncPostgresStore
   
   store = AsyncPostgresStore(
       connection_string="postgresql://user:pass@host:port/db",
       index={
           "dims": 1536,
           "embed": "openai:text-embedding-3-small"
       }
   )
   ```

### Memory Performance
- **With NumPy**: Optimized vector operations for fast similarity search
- **Without NumPy**: Falls back to pure Python (significantly slower)
- **Recommendation**: Always install NumPy for production use

## Usage Examples

### Testing Memory
```bash
# Test memory within single session
python test_memory.py

# Test persistence across restarts
python test_persistence.py

# Interactive testing
python main.py
# Enter user ID: test123
# Share preferences, restart application, and test recall with same user ID
```

### Memory Behavior
- **Single Session**: Memory persists throughout the conversation with fast access
- **Application Restart**: Conversation history and memories are restored from SQLite
- **User Switching**: Different user IDs maintain separate memories
- **Performance**: NumPy-optimized vector operations for semantic search

## Best Practices

1. **User ID Management**: Use consistent, unique user identifiers
2. **Information Richness**: Encourage users to share detailed preferences
3. **Regular Updates**: Allow users to update their stored preferences
4. **Privacy First**: Never store sensitive financial account information
5. **Educational Context**: Always maintain disclaimer about educational purposes