from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
load_dotenv()
url=os.getenv('QDRANT_URL')
api_key=os.getenv('QDRANT_API_KEY')
client=QdrantClient(
    url=url,
    api_key=api_key,
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
loader1 = PyPDFLoader("sam-a16.pdf")
loader2 = PyPDFLoader("sam-s25.pdf")
loader3 = PyPDFLoader("sam-fold.pdf")
docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()
docs = docs1 + docs2 + docs3
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
client.create_collection(
    collection_name="multidoc-rag-agent",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
print(f"Split blog post into {len(all_splits)} sub-documents.")
vector_store = QdrantVectorStore(client=client, embedding=embeddings, collection_name="multidoc-rag-agent")
vector_store.add_documents(all_splits)
print("Documents stored in Qdrant.")