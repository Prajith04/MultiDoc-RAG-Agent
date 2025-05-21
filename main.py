"""
main.py - Command-line interface for the RAG agent.

This file provides a simple terminal-based interaction with the RAG agent,
allowing users to enter queries and receive responses in a continuous loop.
"""
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from agents import rag_agent
from tools import retrieve_tool, calculator_tool

# Initialize chat history to maintain conversation context
chat_history = ChatMessageHistory()

# Create the agent executor with the RAG agent and tools
agent_executor = AgentExecutor(
    agent=rag_agent(),
    tools=[retrieve_tool(), calculator_tool()], 
    verbose=True  # Enable verbose output to see agent's reasoning
)

# Start interactive loop for user queries
while True:
    # Get user input and process through agent
    response = agent_executor.invoke(
        {"input": input("Enter the query:"), "chat_history": chat_history.messages}
    )
    
    # Update chat history with the exchange
    chat_history.add_ai_message(response['input'])
    chat_history.add_ai_message(response['output'])
    
    # Display the complete response object
    print(response)