"""
prompts.py - Defines prompt templates used by the RAG system.

This file provides structured prompts to guide the language model's responses,
including system instructions and formatting for user queries.
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain import hub

def retriever_prompt():
    """
    Creates a prompt template for the retriever component.
    
    This prompt instructs the model to use the provided context (retrieved documents)
    to answer the user's question, and includes information about the available manuals.
    
    Returns:
        ChatPromptTemplate: A formatted prompt template for the retriever
    """
    return ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Use the context to answer the question:\nContext: {context}"
        "these are the titles of manuals you have:\nManuals: {docs}"
    ),
    HumanMessagePromptTemplate.from_template("{query}"),
])  

def agent_prompt():
    """
    Loads a pre-configured prompt for the agent from LangChain Hub.
    
    Uses the OpenAI functions agent prompt which is designed to work effectively
    with tool-calling agents. This prompt helps the agent determine when to use tools
    and how to respond to user queries.
    
    Returns:
        PromptTemplate: A pre-configured prompt template for the agent
    """
    prompt = hub.pull("hwchase17/openai-functions-agent")
    return prompt

