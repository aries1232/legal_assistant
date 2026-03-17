import os
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from chroma_client import get_chroma_client

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_documents")
DEBUG_LOGS = os.getenv("DEBUG_LOGS", "false").lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger(__name__)
if DEBUG_LOGS:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LegalAssistant:
    def __init__(self) -> None:
        self.llm: Optional[GoogleGenAI] = None
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.db = None
        self.chroma_collection = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None

    def _ensure_ready(self) -> None:
        if self.llm is None:
            self.llm = GoogleGenAI(
                model=GOOGLE_LLM_MODEL,
                api_key=GOOGLE_API_KEY,
                max_tokens=1024,
                temperature=0.1,
            )
        if self.embed_model is None:
            self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        if self.db is None:
            self.db = get_chroma_client()
        if self.chroma_collection is None:
            self.chroma_collection = self.db.get_or_create_collection(CHROMA_COLLECTION)
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        if self.index is None:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
            )

    def _to_chat_messages(
        self, chat_history: Optional[List[Dict[str, Any]]]
    ) -> List[ChatMessage]:
        if not chat_history:
            return []
        messages: List[ChatMessage] = []
        for msg in chat_history:
            role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.get("content", "")))
        return messages

    def _build_filters(self, selected_doc_ids: List[str]) -> Optional[MetadataFilters]:
        if not selected_doc_ids:
            return None
        filters = [
            MetadataFilter(key="doc_id", value=doc_id, operator=FilterOperator.EQ)
            for doc_id in selected_doc_ids
        ]
        return MetadataFilters(filters=filters, condition=FilterCondition.OR) # type: ignore

    def _get_chat_engine(self, selected_doc_ids: List[str]) -> CondenseQuestionChatEngine:
        self._ensure_ready()
        retriever = self._build_retriever(selected_doc_ids)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self.llm,
        )
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            llm=self.llm,
            verbose=False,
        )

    def _build_retriever(self, selected_doc_ids: List[str]) -> VectorIndexRetriever:
        self._ensure_ready()
        filters = self._build_filters(selected_doc_ids)
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
            filters=filters,
        )

    def preview_chunks(
        self, question: str, selected_doc_ids: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        retriever = self._build_retriever(selected_doc_ids)
        nodes = retriever.retrieve(question)

        if not nodes and selected_doc_ids:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=None,
            )
            nodes = retriever.retrieve(question)

        previews: List[Dict[str, Any]] = []
        for node in nodes[:top_k]:
            text = (node.node.text or "").strip() # type: ignore
            previews.append(
                {
                    "filename": node.node.metadata.get("filename", "Unknown"),
                    "doc_id": node.node.metadata.get("doc_id", ""),
                    "score": float(getattr(node, "score", 0.0) or 0.0),
                    "text_preview": (text[:300] + "...") if len(text) > 300 else text,
                }
            )
        return previews

    def query(
        self,
        question: str,
        selected_doc_ids: List[str],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        try:
            if DEBUG_LOGS:
                logger.info(
                    "query start | selected_doc_ids=%d | question=%s",
                    len(selected_doc_ids),
                    question[:120],
                )
            retriever = self._build_retriever(selected_doc_ids)
            probe_nodes = retriever.retrieve(question)
            if DEBUG_LOGS:
                logger.info("query probe | filtered_nodes=%d", len(probe_nodes))

            if not probe_nodes and selected_doc_ids:
                retriever = self._build_retriever([])
                probe_nodes = retriever.retrieve(question)
                if DEBUG_LOGS:
                    logger.info("query probe fallback(all docs) | nodes=%d", len(probe_nodes))

            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=self.llm,
            )
            response = query_engine.query(
                QueryBundle(query_str=question, custom_embedding_strs=[question])
            )
            sources: List[Dict[str, Any]] = []
            if hasattr(response, "source_nodes"):
                for source_node in response.source_nodes:
                    filename = source_node.node.metadata.get("filename", "Unknown")
                    doc_id = source_node.node.metadata.get("doc_id", "")
                    score = source_node.score if hasattr(source_node, "score") else 0.0
                    preview = source_node.node.text[:140] + "..." # type: ignore
                    sources.append(
                        {
                            "filename": filename,
                            "doc_id": doc_id,
                            "score": float(score or 0),
                            "text_preview": preview,
                        }
                    )
            answer = str(response or "").strip()
            if DEBUG_LOGS:
                logger.info("query done | sources=%d | answer_len=%d", len(sources), len(answer))
            return {"answer": answer, "sources": sources}
        except Exception as e:
            if DEBUG_LOGS:
                logger.exception("query failed")
            return {"answer": f"Error: {e}", "sources": []}

    def stream_query(
        self,
        question: str,
        selected_doc_ids: List[str],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ):
        if DEBUG_LOGS:
            logger.info(
                "stream start | selected_doc_ids=%d | question=%s",
                len(selected_doc_ids),
                question[:120],
            )
        retriever = self._build_retriever(selected_doc_ids)
        probe_nodes = retriever.retrieve(question)
        if DEBUG_LOGS:
            logger.info("stream probe | filtered_nodes=%d", len(probe_nodes))

        effective_doc_ids = selected_doc_ids
        if not probe_nodes and selected_doc_ids:
            effective_doc_ids = []
            if DEBUG_LOGS:
                logger.info("stream fallback to all docs due to zero filtered nodes")

        chat_engine = self._get_chat_engine(effective_doc_ids)
        llama_history = self._to_chat_messages(chat_history)
        return chat_engine.stream_chat(question, chat_history=llama_history)
