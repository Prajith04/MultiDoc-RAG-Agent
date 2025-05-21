"""
agents.py - Configures and creates the RAG (Retrieval-Augmented Generation) agent.

This file defines the creation of an LLM-powered agent that can use tools to retrieve 
information from a vector database and perform calculations.
"""
from langchain.agents import create_tool_calling_agent
from query_vectordb import chat_model
from tools import retrieve_tool, calculator_tool
from prompts import agent_prompt

def rag_agent():
    """
    Creates and configures a tool-calling agent for RAG functionality.
    
    This function:
    1. Initializes the language model (LLM) from query_vectordb.py
    2. Sets up the available tools (retriever and calculator)
    3. Configures the agent with the appropriate prompt
    4. Returns the configured agent ready to process queries
    
    Returns:
        Agent: A configured LangChain agent with retrieval capabilities
    """
    llm=chat_model()
    tools = [retrieve_tool(), calculator_tool()]
    prompt=agent_prompt()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

