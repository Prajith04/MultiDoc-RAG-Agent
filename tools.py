"""
tools.py - Defines the tools available to the RAG agent.

This file creates specialized tools that the agent can use to:
1. Search the vector database for relevant Samsung device documentation
2. Perform mathematical calculations when needed
"""
from langchain.tools.retriever import create_retriever_tool
from query_vectordb import chat_model, init_vector_store, small_chat_model
from langchain_community.agent_toolkits.load_tools import load_tools

def retrieve_tool():
    """
    Creates a retriever tool for searching Samsung-related documentation.
    
    This function:
    1. Initializes the vector database connection
    2. Sets up a retriever with similarity search parameters
    3. Creates a tool with description for the agent to understand when to use it
    
    Returns:
        Tool: A configured retriever tool for Samsung documentation lookup
    """
    doc_store=init_vector_store()
    retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,})
    retriever_tool = create_retriever_tool(
    retriever,
    "VectorDB_search",
    "Use this tool when you need to answer questions about Samsung mobile phones, including their features, settings, or troubleshooting. For example: how to enable dark mode, battery saving tips, or camera settings.",)
    return retriever_tool

def calculator_tool():
    """
    Creates a mathematical calculation tool.
    
    Uses the LangChain llm-math tool powered by a smaller LLM to handle
    mathematical operations requested by users.
    
    Returns:
        Tool: A configured calculator tool for mathematical operations
    """
    return load_tools(["llm-math"], llm=small_chat_model())[0]