import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

def get_retriever():
    # 1. Setup Embeddings (Must match the one used in ingestion)
    embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Connect to Qdrant
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )
    
    # 3. Create the retriever 
    # 'k=5' means it will pull the top 5 most relevant code snippets
    return vectorstore.as_retriever(search_kwargs={"k": 5})