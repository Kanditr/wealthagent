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
- **Real portfolio integration** with Alpaca brokerage - access live portfolio data, positions, and performance
- **Personalized experience** - each user gets their own memory namespace for privacy
- **Intelligent memory management** - automatically extracts and stores important financial information
- **Memory search capabilities** - retrieves relevant context from past conversations
- **Portfolio analysis** - provides advice based on actual holdings and account information
- **OpenAI ChatGPT integration** with gpt-4o-mini model
- **Optional LangSmith tracing** for debugging and monitoring

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `ALPACA_API_KEY` | No | Your Alpaca API key for portfolio access |
| `ALPACA_SECRET_KEY` | No | Your Alpaca secret key |
| `ALPACA_PAPER_TRADING` | No | Enable paper trading mode (true/false, default: true) |
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

### Portfolio Features (Optional)
With Alpaca API setup, the agent can:
- Access your real portfolio positions and holdings
- Provide account summary and performance data
- Give personalized advice based on your actual investments
- Track portfolio changes and performance over time
- Analyze individual positions and suggest optimizations

### Testing
Run the test scripts to verify functionality:
```bash
# Test memory within single session
python test_memory.py

# Test persistence across application restarts
python test_persistence.py

# Test Alpaca portfolio integration
python test_alpaca.py
```

### Memory Persistence
✅ **Automatic Persistence**: The agent now uses SQLite + InMemoryStore hybrid:
- **Conversation history** persists across application restarts
- **User memories** are automatically saved to SQLite database
- **Fast performance** during active sessions with InMemoryStore
- **User isolation** - each user gets their own secure memory space

A `wealth_agent_memories.db` file will be created automatically in your project directory.

See `MEMORY.md` for detailed memory system documentation.

## Alpaca Setup (Optional)

To enable portfolio features:

1. **Sign up for Alpaca**: Visit [alpaca.markets](https://alpaca.markets/) and create an account
2. **Get API keys**: Go to Paper Trading section and generate API key and secret
3. **Add to .env**: Copy credentials to your `.env` file
4. **Test connection**: Run `python test_alpaca.py` to verify setup

The agent works without Alpaca - portfolio features are optional but provide much more personalized advice.

**Note:** All responses are for educational purposes only and not financial advice.