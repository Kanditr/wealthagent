# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wealth management agent chat application built with LangChain, LangGraph, LangMem, and OpenAI ChatGPT. The agent uses advanced memory capabilities to provide personalized financial guidance by remembering user preferences, goals, and past conversations across sessions.

## Development Setup

### Quick Setup
```bash
# Automated setup with virtual environment
./setup.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application
```bash
# Start the chat interface (prompts for user ID)
python main.py

# Test memory functionality
python test_memory.py
```

### Environment Configuration
Required in `.env` file:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGSMITH_TRACING=true` (optional): Enable tracing
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional)
- `LANGSMITH_PROJECT`: Project name for LangSmith (optional)

## Architecture Considerations

When developing this wealth agent application, consider:

### Core Components
- **Chat Interface**: User interaction layer for financial queries
- **LLM Integration**: LangChain-based conversation management
- **Financial Data Sources**: Integration with market data APIs
- **Risk Assessment**: Portfolio analysis and risk evaluation modules
- **Recommendation Engine**: Investment advice generation
- **User Authentication**: Secure user session management
- **Data Storage**: Conversation history and user preferences

### Security Requirements
- Never log or store sensitive financial information
- Implement proper API key management
- Use environment variables for configuration
- Validate all user inputs for financial data
- Implement rate limiting for API calls

### LangChain Integration Patterns
- **LangMem SDK**: Long-term memory for user preferences and conversation history
- **User namespacing**: Each user gets isolated memory space for privacy
- **Memory tools**: `create_manage_memory_tool` and `create_search_memory_tool` for automatic memory management
- **React agent**: Uses `create_react_agent` for tool integration
- **InMemoryStore**: Current storage backend (can be upgraded to persistent storage)
- Implement message trimming for token management
- Create custom tools for financial calculations
- Use prompt templates for consistent financial advice formatting
- Implement streaming for real-time responses

## File Structure Expectations

Once development begins, expect:
```
/
├── src/ or app/              # Main application code
├── tests/                    # Test files
├── config/                   # Configuration files
├── data/                     # Static data or schemas
├── docs/                     # Documentation
├── .env.example              # Environment variables template
└── requirements.txt or package.json  # Dependencies
```

## Financial Domain Considerations

- Implement disclaimer mechanisms for investment advice
- Ensure compliance with financial regulations
- Use proper financial calculation libraries
- Implement data validation for financial inputs
- Consider real-time vs. delayed market data requirements