#!/usr/bin/env python3
"""
Test script for persistent memory across application restarts
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WealthAgentChat

load_dotenv()

def test_memory_persistence():
    print("Testing Memory Persistence Across Restarts")
    print("=" * 50)
    
    user_id = "persistent_test_user"
    thread_id = "test_session"
    
    print(f"Using User ID: {user_id}")
    print(f"Using Thread ID: {thread_id}")
    
    # Create agent instance
    agent = WealthAgentChat(user_id=user_id)
    
    print("\n--- Session 1: Storing Information ---")
    
    # Store personal info
    response1 = agent.chat("Hi! My name is Bob and I'm 35 years old.", thread_id)
    print(f"Stored name/age: {response1}")
    
    # Store preferences
    response2 = agent.chat("I'm a moderate risk investor. I like a mix of stocks and bonds, about 70/30 allocation.", thread_id)
    print(f"Stored preferences: {response2}")
    
    # Store goals
    response3 = agent.chat("My goal is to save $200,000 for retirement by age 60. I also want to buy a vacation home.", thread_id)
    print(f"Stored goals: {response3}")
    
    print("\n--- Testing Immediate Recall (Same Session) ---")
    
    # Test immediate recall
    response4 = agent.chat("What's my name and age?", thread_id)
    print(f"Name recall: {response4}")
    
    response5 = agent.chat("What's my investment allocation preference?", thread_id)
    print(f"Preference recall: {response5}")
    
    print("\n--- Simulating Application Restart ---")
    print("Creating new agent instance (simulates restart)...")
    
    # Create NEW agent instance (simulates restart)
    agent_restart = WealthAgentChat(user_id=user_id)
    
    print("\n--- Session 2: Testing Persistence After 'Restart' ---")
    
    # Test if conversation history persists
    response6 = agent_restart.chat("What's my name?", thread_id)
    print(f"Name after restart: {response6}")
    
    response7 = agent_restart.chat("What are my investment preferences?", thread_id)
    print(f"Preferences after restart: {response7}")
    
    response8 = agent_restart.chat("What are my financial goals?", thread_id)
    print(f"Goals after restart: {response8}")
    
    response9 = agent_restart.chat("Can you summarize everything you know about me?", thread_id)
    print(f"Complete summary: {response9}")
    
    print("\n" + "=" * 50)
    print("Persistence test completed!")
    
    # Check if database file was created
    db_file = "wealth_agent_memories.db"
    if os.path.exists(db_file):
        print(f"✅ SQLite database created: {db_file}")
        print(f"📁 Database size: {os.path.getsize(db_file)} bytes")
    else:
        print("❌ SQLite database not found")

if __name__ == "__main__":
    try:
        test_memory_persistence()
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")