"""
query_vectordb.py - Contains functions for initializing language models and managing vector database interactions.

This file provides core functionality for:
1. Initializing different language models for the RAG system
2. Setting up connections to the vector database
3. Retrieving relevant documents based on user queries
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()

def chat_model():
    """
    Initialize the primary language model for detailed responses.
    
    Uses the Groq API with the Mistral-SABA-24B model for high-quality responses.
    
    Returns:
        LLM: A configured language model instance
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = init_chat_model("mistral-saba-24b", model_provider="groq",api_key=groq_api_key)
    return llm

def small_chat_model():
    """
    Initialize a smaller language model for less complex tasks.
    
    Uses the Llama-3.3-70B-Versatile model via Groq API for tasks like calculation.
    
    Returns:
        LLM: A configured language model instance optimized for simpler tasks
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq",api_key=groq_api_key)
    return llm

def init_vector_store():
    """
    Initialize the connection to the Qdrant vector database.
    
    Sets up the embedding model and connects to the existing collection in Qdrant
    that contains the document embeddings.
    
    Returns:
        QdrantVectorStore: A vector store instance connected to the document collection
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="multidoc-rag-agent",
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'))
    return doc_store

def retrieve_docs(query, doc_store):
    """
    Retrieve relevant documents from the vector database based on the user query.
    
    Args:
        query (str): The user's search query
        doc_store (VectorStore): The initialized vector store
    
    Returns:
        list: A list of retrieved document chunks relevant to the query
    """
    retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,})
    response=retriever.invoke(query)
    return response


