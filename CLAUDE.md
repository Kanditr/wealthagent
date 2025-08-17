# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wealth management agent chat application project. The codebase is currently empty and ready for initial development.

## Development Setup

Since this is a new project, the following setup will likely be needed:

### For Python-based LangChain Implementation
```bash
# Install dependencies
pip install langchain-core langgraph
pip install "langchain[google-genai]"  # or preferred LLM provider
pip install streamlit  # for web interface
pip install python-dotenv  # for environment variables

# Run the application
python main.py
# or for Streamlit web app
streamlit run app.py

# Run tests
pytest
# or
python -m pytest tests/
```

### For Node.js Implementation
```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build production
npm run build

# Run tests
npm test
npm run test:watch  # for watch mode

# Linting and formatting
npm run lint
npm run format
```

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
- Use StateGraph for conversation state management
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