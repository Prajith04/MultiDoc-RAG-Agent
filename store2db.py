"""
store2db.py - Loads PDF documents and stores them in a Qdrant vector database.

This file:
1. Handles document loading from PDF files
2. Splits documents into manageable chunks
3. Creates vector embeddings for each chunk
4. Stores the embeddings in a Qdrant vector database for retrieval
"""
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client with credentials from environment variables
url=os.getenv('QDRANT_URL')
api_key=os.getenv('QDRANT_API_KEY')
client=QdrantClient(
    url=url,
    api_key=api_key,
)

# Initialize the embedding model for document vectorization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load PDF documents using LangChain's PyPDFLoader
loader1 = PyPDFLoader("sam-a16.pdf")  # Samsung A16 manual
loader2 = PyPDFLoader("sam-s25.pdf")  # Samsung S25 manual
loader3 = PyPDFLoader("sam-fold.pdf") # Samsung Fold manual
docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()
docs = docs1 + docs2 + docs3  # Combine all documents

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Set chunk size to 1000 characters
    chunk_overlap=200,  # 200 character overlap between chunks
    add_start_index=True,  # Track the starting index in the original document
)
all_splits = text_splitter.split_documents(docs)

# Create the vector collection in Qdrant
client.create_collection(
    collection_name="multidoc-rag-agent",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),  # Configuration based on the embedding model
)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Initialize vector store and add document chunks
vector_store = QdrantVectorStore(client=client, embedding=embeddings, collection_name="multidoc-rag-agent")
vector_store.add_documents(all_splits)

print("Documents stored in Qdrant.")