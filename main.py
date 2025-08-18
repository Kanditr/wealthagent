import os
import sqlite3
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langmem import create_manage_memory_tool, create_search_memory_tool

load_dotenv()

class WealthAgentChat:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        
        # SQLite for conversation persistence + InMemoryStore for speed
        self.sqlite_conn = sqlite3.connect("wealth_agent_memories.db", check_same_thread=False)
        self.memory = SqliteSaver(self.sqlite_conn)
        
        # Configure InMemoryStore with proper embedding settings for LangMem
        self.store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small"
            }
        )
        self.memory_namespace = ("memories", user_id)
        
        # Create agent with persistent memory and fast in-session storage
        self.agent = create_react_agent(
            ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
            tools=[
                create_manage_memory_tool(namespace=self.memory_namespace),
                create_search_memory_tool(namespace=self.memory_namespace),
            ],
            store=self.store,
            checkpointer=self.memory
        )
    
    def chat(self, message: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": f"{self.user_id}_{thread_id}"}}
        
        # Create system message for wealth management context with explicit memory instructions
        system_msg = SystemMessage(
            content="You are a helpful wealth management assistant with memory capabilities. "
                    "IMPORTANT: Actively use your memory tools to:\n"
                    "1. STORE important user information (name, preferences, goals, risk tolerance, etc.)\n"
                    "2. SEARCH your memory before responding to find relevant past conversations\n"
                    "3. When users share personal info, use the manage_memory tool to save it\n"
                    "4. When users ask questions, use search_memory tool to find relevant context\n"
                    "Always remember: You have memory tools - use them proactively!\n"
                    "Provide clear, accurate financial guidance while noting this is for educational purposes only."
        )
        
        response = self.agent.invoke(
            {"messages": [system_msg, HumanMessage(content=message)]}, 
            config
        )
        
        return response["messages"][-1].content
    
    def close(self):
        """Close the SQLite connection"""
        if hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()

def main():
    # Get user ID for personalized memory
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    chat_agent = WealthAgentChat(user_id=user_id)
    
    print(f"Wealth Agent Chat (User: {user_id}) - Type 'quit' to exit")
    print("The agent will remember your preferences across conversations!")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Your preferences have been saved.")
            break
        
        if not user_input:
            continue
        
        try:
            response = chat_agent.chat(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Clean up
    chat_agent.close()

if __name__ == "__main__":
    main()