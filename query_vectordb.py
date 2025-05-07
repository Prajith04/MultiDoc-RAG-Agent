from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
def chat_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = init_chat_model("mistral-saba-24b", model_provider="groq",api_key=groq_api_key)
    return llm
def small_chat_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq",api_key=groq_api_key)
    return llm
def init_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="multidoc-rag-agent",
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'))
    return doc_store
def retrieve_docs(query, doc_store):
    retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,})
    response=retriever.invoke(query)
    return response


