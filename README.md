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

- Interactive chat interface
- Memory persistence across conversations
- LangGraph state management
- OpenAI ChatGPT integration
- Optional LangSmith tracing

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (true/false) |
| `LANGSMITH_API_KEY` | No | Your LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

## Usage

The chat agent provides financial guidance and wealth management assistance. All responses are for educational purposes only.