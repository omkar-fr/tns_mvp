import os
from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def ingest_repo(repo_path: str):
    """
    Ingests a local directory into Qdrant. 
    Wipes the existing collection to ensure only the current project is active.
    """
    print(f"🔍 Loading code from: {repo_path}")
    
    # 1. Load the files
    loader = GenericLoader.from_filesystem(
        repo_path, 
        glob="**/*", 
        suffixes=[".py"], 
        parser=LanguageParser()
    )
    docs = loader.load()
    
    if not docs:
        print("⚠️ No python files found in the selected directory.")
        return

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=1000, 
        chunk_overlap=100
    )
    texts = splitter.split_documents(docs)

    # 3. Initialize Local Embeddings
    print("🧠 Initializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Store in Qdrant
    print(f"🚀 Indexing {len(texts)} chunks into Qdrant...")
    QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url=os.getenv("QDRANT_URL"),
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        force_recreate=True  # This is crucial: it wipes the old project data
    )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    # Default behavior for terminal run
    ingest_repo("./data")