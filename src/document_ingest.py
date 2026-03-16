import os
import tempfile
import uuid
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from chroma_client import get_chroma_client

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_documents")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))


class DocumentIngestor:
    def __init__(self) -> None:
        self.db = get_chroma_client()
        self.collection = self.db.get_or_create_collection(CHROMA_COLLECTION)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        self.converter = DocumentConverter()
        self.parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )

    def _extract_text(self, pdf_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        try:
            result = self.converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def ingest_pdf(self, filename: str, pdf_bytes: bytes, doc_id: str) -> Dict[str, Any]:
        try:
            extracted_text = self._extract_text(pdf_bytes)
            if not extracted_text.strip():
                return {
                    "ok": False,
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_count": 0,
                    "error": "No extractable text found in PDF.",
                }

            doc = Document(
                text=extracted_text,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "source_type": "pdf",
                },
            )
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            index = VectorStoreIndex.from_documents(
                [doc],
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=[self.parser],
                show_progress=False,
            )
            nodes = self.parser.get_nodes_from_documents([doc])
            return {
                "ok": True,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": len(nodes),
                "error": None,
            }
        except Exception as e:
            return {
                "ok": False,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": 0,
                "error": str(e),
            }


def generate_doc_id() -> str:
    return str(uuid.uuid4())
