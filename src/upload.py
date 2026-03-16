import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

try:
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from chroma_client import get_chroma_client

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "25"))
CHROMA_INSERT_BATCH_SIZE = int(os.getenv("CHROMA_INSERT_BATCH_SIZE", "500"))
CHUNK_SIZE_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 200

def load_documents():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "crawled_data"
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist. Please run crawl.py first.")
        return []

    manifest_path = data_dir / "manifest.json"
    manifest_records = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_records = {
                item["filename"]: item
                for item in json.load(f)
                if item.get("filename")
            }

    documents = []
    
    for file_path in sorted(data_dir.glob("*.md")):
        filename = file_path.name
        print(f"Loading document from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        manifest_record = manifest_records.get(filename, {})
        url = manifest_record.get("url")
        if not url:
            url_part = filename.replace(".md", "").replace("_", "/")
            url = f"https://{url_part}"

        metadata = {
            "url": url,
            "filename": filename,
        }
        if manifest_record.get("scope"):
            metadata["scope"] = manifest_record["scope"]
        if manifest_record.get("title"):
            metadata["title"] = manifest_record["title"]

        doc = Document(text=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def ingest_to_chroma(documents):
    if not documents:
        print("No documents were loaded.")
        return

    db = get_chroma_client()
        
    chroma_collection = db.get_or_create_collection("gitlab_handbook")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    
     
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    
    print(f"Building/updating index for {len(documents)} documents using {CHUNK_SIZE_TOKENS} chunk size...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        insert_batch_size=CHROMA_INSERT_BATCH_SIZE,
        transformations=[parser],
        show_progress=True
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    docs = load_documents()
    ingest_to_chroma(docs)
