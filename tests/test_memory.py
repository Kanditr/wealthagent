#!/usr/bin/env python3
"""
Test script for LangMem memory functionality
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WealthAgentChat

load_dotenv()

def test_memory_functionality():
    print("Testing LangMem Memory Functionality")
    print("=" * 50)
    
    # Create agent for test user
    agent = WealthAgentChat(user_id="test_user_123")
    
    # Test 1: Store personal information
    print("\n1. Storing personal information...")
    response1 = agent.chat("Hi! My name is Alice and I'm 28 years old.")
    print(f"Response: {response1}")
    
    # Test 2: Store investment preferences
    print("\n2. Storing investment preferences...")
    response2 = agent.chat("I'm a conservative investor who prefers low-risk investments like bonds and index funds. I don't like volatile stocks.")
    print(f"Response: {response2}")
    
    # Test 3: Store financial goals
    print("\n3. Storing financial goals...")
    response3 = agent.chat("My goal is to save $100,000 for retirement by age 65. I want to buy a house in the next 5 years too.")
    print(f"Response: {response3}")
    
    # Test 4: Test name retrieval
    print("\n4. Testing name retrieval...")
    response4 = agent.chat("What's my name?")
    print(f"Response: {response4}")
    
    # Test 5: Test preference retrieval
    print("\n5. Testing investment preference retrieval...")
    response5 = agent.chat("What do you know about my investment preferences?")
    print(f"Response: {response5}")
    
    # Test 6: Test comprehensive memory
    print("\n6. Testing comprehensive memory recall...")
    response6 = agent.chat("Can you summarize everything you know about me?")
    print(f"Response: {response6}")
    
    # Test 7: Test memory-based recommendations
    print("\n7. Testing memory-based investment advice...")
    response7 = agent.chat("Can you recommend some investments for me based on what you know?")
    print(f"Response: {response7}")
    
    print("\n" + "=" * 50)
    print("Memory test completed!")
    print("Note: If memory isn't working, check that OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    try:
        test_memory_functionality()
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")