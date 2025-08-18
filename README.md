# Wealth Agent Chat

A simple chatbot for wealth management assistance built with LangChain, LangGraph, and OpenAI ChatGPT.

## Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script (creates venv and installs dependencies)
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the chat application
python main.py
```

## Features

- **Interactive chat interface** with user identification
- **Long-term memory** using LangMem SDK - remembers user preferences, financial goals, and past conversations
- **Personalized experience** - each user gets their own memory namespace for privacy
- **Intelligent memory management** - automatically extracts and stores important financial information
- **Memory search capabilities** - retrieves relevant context from past conversations
- **OpenAI ChatGPT integration** with gpt-4o-mini model
- **Optional LangSmith tracing** for debugging and monitoring

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (true/false) |
| `LANGSMITH_API_KEY` | No | Your LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

## Usage

### Basic Usage
1. Run `python main.py`
2. Enter a user ID (or press Enter for 'demo_user')
3. Start chatting! The agent will remember your preferences and past conversations

### Memory Features
The agent automatically:
- Stores your investment preferences and risk tolerance
- Remembers your financial goals and timeline
- Recalls past conversations and advice given
- Provides personalized recommendations based on your history

### Testing Memory
Run the test scripts to see memory functionality:
```bash
# Test memory within single session
python test_memory.py

# Test persistence across application restarts
python test_persistence.py
```

### Memory Persistence
✅ **Automatic Persistence**: The agent now uses SQLite + InMemoryStore hybrid:
- **Conversation history** persists across application restarts
- **User memories** are automatically saved to SQLite database
- **Fast performance** during active sessions with InMemoryStore
- **User isolation** - each user gets their own secure memory space

A `wealth_agent_memories.db` file will be created automatically in your project directory.

See `MEMORY.md` for detailed memory system documentation.

**Note:** All responses are for educational purposes only and not financial advice.