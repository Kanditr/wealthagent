import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

class WealthAgentChat:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.memory = MemorySaver()
        self.app = self._create_workflow()
    
    def _create_workflow(self):
        workflow = StateGraph(state_schema=MessagesState)
        
        def call_model(state: MessagesState):
            system_prompt = SystemMessage(
                content="You are a helpful wealth management assistant. "
                        "Provide clear, accurate financial guidance while being "
                        "careful to note that this is for educational purposes only."
            )
            
            messages = [system_prompt] + state["messages"]
            response = self.model.invoke(messages)
            return {"messages": [response]}
        
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, message: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]
        
        output = self.app.invoke({"messages": input_messages}, config)
        return output["messages"][-1].content

def main():
    chat_agent = WealthAgentChat()
    
    print("Wealth Agent Chat - Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = chat_agent.chat(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()