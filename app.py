import os
import re
import unicodedata
from io import BytesIO

import streamlit as st
import pypdfium2 as pdfium

try:
    from src.document_ingest import DocumentIngestor, generate_doc_id
    from src.legal_bot import LegalAssistant
    from src.supabase_client import supabase_db
except ModuleNotFoundError:
    from document_ingest import DocumentIngestor, generate_doc_id
    from legal_bot import LegalAssistant
    from supabase_client import supabase_db

st.set_page_config(page_title="Legal Assistant", page_icon=" ", layout="wide")
st.title("Legal Assistant")
st.markdown(
    """
    <style>
    :root {
        --chat-shell-width: min(820px, calc(100vw - 3rem));
        --chat-input-bg: rgb(38, 39, 48);
    }
    div.stButton > button {
        white-space: nowrap;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-bottom: 9rem;
    }
    div[data-testid="stChatInput"] {
        position: fixed;
        left: 50%;
        bottom: 1.25rem;
        transform: translateX(-50%);
        width: var(--chat-shell-width);
        z-index: 999;
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
        backdrop-filter: none;
    }
    div[data-testid="stChatInput"] > div {
        background: var(--chat-input-bg);
        border-radius: 18px;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 1rem;
    }
    div[data-testid="stChatMessage"] {
        animation: fadeSlideIn 220ms ease-out;
    }
    @keyframes fadeSlideIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sanitize_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._")
    return sanitized or "uploaded.pdf"


def format_size(num_bytes: int) -> str:
    value = float(num_bytes or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def render_pdf_thumbnail(pdf_bytes: bytes):
    try:
        pdf = pdfium.PdfDocument(BytesIO(pdf_bytes))
        page = pdf[0]
        bitmap = page.render(scale=0.35) # type: ignore
        pil_image = bitmap.to_pil()
        page.close()
        pdf.close()
        return pil_image
    except Exception:
        return None


@st.cache_resource
def get_assistant() -> LegalAssistant:
    return LegalAssistant()


@st.cache_resource
def get_ingestor() -> DocumentIngestor:
    return DocumentIngestor()

def load_assistant() -> tuple[LegalAssistant | None, str | None]:
    try:
        return get_assistant(), None
    except Exception as exc:
        return None, str(exc)


def load_ingestor() -> tuple[DocumentIngestor | None, str | None]:
    try:
        return get_ingestor(), None
    except Exception as exc:
        return None, str(exc)


storage_bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "legal-docs")
debug_logs = os.getenv("DEBUG_LOGS", "false").lower() in {"1", "true", "yes", "on"}

if "session_id" not in st.session_state:
    st.session_state.session_id = supabase_db.create_session()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preview_doc_id" not in st.session_state:
    st.session_state.preview_doc_id = None
if "chat_locked_doc_ids" not in st.session_state:
    st.session_state.chat_locked_doc_ids = None
if "_latest_sources" not in st.session_state:
    st.session_state["_latest_sources"] = None
if "_latest_debug" not in st.session_state:
    st.session_state["_latest_debug"] = None

tab_library, tab_chat = st.tabs(["Document Library", "Legal Assistant"])

with tab_library:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    has_files = bool(uploaded_files)

    if st.button("Upload", disabled=not has_files, type="primary" if has_files else "secondary"):
        if not uploaded_files:
            st.info("Select at least one PDF.")
        else:
            ingestor, ingestor_error = load_ingestor()
            if ingestor is None:
                st.error(f"Document ingestion is unavailable: {ingestor_error}")
                st.stop()
            for uploaded_file in uploaded_files:
                st.markdown(f"**Processing: {uploaded_file.name}**")
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.caption("Initializing document context...")
                progress_bar.progress(5)

                doc_id = generate_doc_id()
                safe_filename = sanitize_filename(uploaded_file.name)
                file_path = f"docs/{doc_id}/{safe_filename}"
                file_bytes = uploaded_file.getvalue()

                supabase_db.create_document(
                    doc_id=doc_id,
                    filename=safe_filename,
                    file_path=file_path,
                    file_size_bytes=len(file_bytes),
                    session_id=st.session_state.session_id,
                )

                progress_text.caption("Uploading PDF to storage...")
                progress_bar.progress(15)

                uploaded = supabase_db.upload_pdf_to_storage(
                    bucket=storage_bucket,
                    file_path=file_path,
                    content=file_bytes,
                )
                if not uploaded:
                    supabase_db.update_document_status(
                        doc_id=doc_id,
                        ingest_status="failed",
                        ingest_error="Upload to Supabase Storage failed.",
                    )
                    progress_text.error(f"Upload failed: {uploaded_file.name}")
                    continue
                
                def ingest_callback(msg: str, progress: int):
                    # Progress from ingestion starts at 20, map it clearly
                    progress_text.caption(msg)
                    progress_bar.progress(progress)

                result = ingestor.ingest_pdf(
                    filename=uploaded_file.name,
                    pdf_bytes=file_bytes,
                    doc_id=doc_id,
                    status_callback=ingest_callback
                )
                
                if result["ok"]:
                    progress_text.empty()
                    progress_bar.empty()
                    supabase_db.update_document_status(
                        doc_id=doc_id,
                        ingest_status="ready",
                        chunk_count=result["chunk_count"],
                    )
                    st.success(f"Indexed {uploaded_file.name} ({result['chunk_count']} chunks)")
                else:
                    supabase_db.update_document_status(
                        doc_id=doc_id,
                        ingest_status="failed",
                        ingest_error=result["error"],
                    )
                    st.error(f"Failed: {uploaded_file.name} - {result['error']}")

    st.subheader("Document Library")
    docs = supabase_db.list_documents()
    visible_docs = [d for d in docs if d.get("ingest_status") != "failed"]
    if docs:
        total_docs = len(visible_docs)
        ready_count = len([d for d in visible_docs if d.get("ingest_status") == "ready"])
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total", total_docs)
        col_m2.metric("Ready", ready_count)

        search_text = st.text_input("Search documents", placeholder="Type filename...")
        filtered_docs = [
            d
            for d in visible_docs
            if search_text.strip().lower() in str(d.get("filename", "")).lower()
        ]

        library_col, preview_col = st.columns([2.2, 1.1], gap="large")
        with library_col:
            if not filtered_docs:
                st.caption("No matching documents.")
            else:
                grid_cols = st.columns(2)
                for idx, doc in enumerate(filtered_docs):
                    col = grid_cols[idx % 2]
                    with col:
                        doc_id = str(doc.get("id", ""))
                        status = str(doc.get("ingest_status", "unknown")).upper()
                        with st.container(border=True):
                            filename = str(doc.get("filename", "Unknown"))
                            display_name = filename if len(filename) <= 42 else filename[:39] + "..."
                            st.markdown(f"**{display_name}**")
                            st.caption(f"Status: {status}")
                            st.caption(f"Chunks: {doc.get('chunk_count', 0)}")
                            st.caption(f"Size: {format_size(int(doc.get('file_size_bytes') or 0))}")
                            action_cols = st.columns(2)
                            if action_cols[0].button(
                                "Preview", key=f"preview_doc_{doc_id}", use_container_width=True
                            ):
                                st.session_state.preview_doc_id = doc_id
                            if action_cols[1].button(
                                "Delete", key=f"delete_doc_{doc_id}", use_container_width=True
                            ):
                                deleted = supabase_db.soft_delete_document(doc_id)
                                file_path = str(doc.get("file_path") or "")
                                storage_removed = True
                                if file_path:
                                    storage_removed = supabase_db.delete_pdf_from_storage(
                                        bucket=storage_bucket,
                                        file_path=file_path,
                                    )
                                if deleted:
                                    if storage_removed:
                                        st.success(f"Deleted {doc.get('filename')}")
                                    else:
                                        st.warning(
                                            "Marked deleted in library, but storage delete failed for "
                                            f"{doc.get('filename')}"
                                        )
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete {doc.get('filename')}")

        with preview_col:
            st.markdown("**Document Preview**")
            preview_doc_id = st.session_state.preview_doc_id
            if preview_doc_id:
                preview_doc = next(
                    (d for d in docs if str(d.get("id", "")) == str(preview_doc_id)),
                    None,
                )
                if preview_doc:
                    file_path = str(preview_doc.get("file_path") or "")
                    file_name = str(preview_doc.get("filename") or "document.pdf")
                    pdf_bytes = (
                        supabase_db.download_pdf_from_storage(storage_bucket, file_path)
                        if file_path
                        else None
                    )
                    if pdf_bytes:
                        thumb = render_pdf_thumbnail(pdf_bytes)
                        if thumb is not None:
                            st.image(thumb, caption=file_name, use_container_width=True)
                        else:
                            st.caption("Thumbnail unavailable for this PDF.")
                        st.caption(f"File: {file_name}")
                        st.caption(f"Size: {format_size(int(preview_doc.get('file_size_bytes') or 0))}")
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"download_doc_{preview_doc_id}",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Could not load this document from storage for preview/download.")
                else:
                    st.caption("Select a document to preview.")
            else:
                st.caption("Select a document to preview.")
    else:
        st.caption("No documents uploaded yet.")

with tab_chat:
    chat_left, chat_center, chat_right = st.columns([1.2, 4.6, 1.2])
    with chat_center:
        st.subheader("Chat With Your Documents")
        docs = supabase_db.list_documents()
        ready_docs = [d for d in docs if d.get("ingest_status") == "ready"]
        selected_docs_map = {
            f"{d.get('filename')} ({d.get('id')})": d.get("id") for d in ready_docs
        }

        is_chat_started = len(st.session_state.messages) > 0
        if not is_chat_started:
            selected_labels = st.multiselect(
                "Select documents for this chat",
                options=list(selected_docs_map.keys()),
            )
            selected_doc_ids = [selected_docs_map[label] for label in selected_labels]
            active_doc_ids = selected_doc_ids if selected_doc_ids else list(selected_docs_map.values())
        else:
            selected_doc_ids = []
            locked_doc_ids = st.session_state.chat_locked_doc_ids or []
            active_doc_ids = locked_doc_ids
            st.caption(
                f"Using {len(active_doc_ids)} locked document(s) for this chat. "
                "Start a new chat to change selection."
            )
            if st.button("Start New Chat"):
                st.session_state.session_id = supabase_db.create_session()
                st.session_state.messages = []
                st.session_state.chat_locked_doc_ids = None
                st.session_state["_latest_sources"] = None
                st.session_state["_latest_debug"] = None
                st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Ask me anything about your documents...")
        if query:
            if not active_doc_ids:
                st.warning("No indexed documents available yet. Upload and process a PDF first.")
            else:
                assistant, assistant_error = load_assistant()
                if assistant is None:
                    st.error(f"Chat service is unavailable: {assistant_error}")
                    st.stop()
                if st.session_state.chat_locked_doc_ids is None:
                    st.session_state.chat_locked_doc_ids = active_doc_ids
                st.session_state.messages.append({"role": "user", "content": query})
                supabase_db.save_message(
                    st.session_state.session_id,
                    "user",
                    query,
                    selected_doc_ids=active_doc_ids, # type: ignore
                )

                full_response = ""
                sources = []
                debug_info = {
                    "ready_docs_count": len(ready_docs),
                    "selected_doc_ids_count": len(selected_doc_ids),
                    "active_doc_ids_count": len(active_doc_ids),
                    "stream_used": False,
                    "stream_tokens": 0,
                    "fallback_used": False,
                }

                with st.chat_message("assistant"):
                    status_message = "Searching across relevant documents..."
                    status = st.status(status_message, expanded=False)
                    response_placeholder = st.empty()

                    stream_response = assistant.stream_query(
                        query,
                        selected_doc_ids=active_doc_ids, # type: ignore
                        chat_history=st.session_state.messages[:-1],
                    )
                    if stream_response:
                        debug_info["stream_used"] = True
                        try:
                            for token in stream_response.response_gen:
                                if not full_response:
                                    status.update(
                                        label=status_message,
                                        state="running",
                                    )
                                full_response += token
                                debug_info["stream_tokens"] += 1
                                response_placeholder.markdown(full_response + "▌")
                        except Exception as e:
                            full_response += f"\n\nError during streaming: {e}"

                    normalized_response = full_response.strip().lower()
                    if (not full_response.strip()) or normalized_response == "empty response":
                        debug_info["fallback_used"] = True
                        status.update(
                            label=status_message,
                            state="running",
                        )
                        result = assistant.query(
                            query,
                            selected_doc_ids=active_doc_ids, # type: ignore
                            chat_history=st.session_state.messages[:-1],
                        )
                        full_response = (result.get("answer") or "").strip()
                        if (not full_response) or full_response.lower() == "empty response":
                            full_response = (
                                "No answer could be generated from indexed documents. "
                                "Try a more specific question or re-process the PDF."
                            )
                        sources = result.get("sources") or []
                    elif stream_response and hasattr(stream_response, "source_nodes"):
                        for source_node in stream_response.source_nodes:
                            sources.append(
                                {
                                    "filename": source_node.node.metadata.get("filename", "Unknown"),
                                    "doc_id": source_node.node.metadata.get("doc_id", ""),
                                    "score": float(getattr(source_node, "score", 0.0) or 0.0),
                                    "text_preview": source_node.node.text[:140] + "...",
                                }
                            )

                    response_placeholder.markdown(full_response)
                    status.update(label="Answer ready", state="complete")

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response, "sources": sources}
                )
                supabase_db.save_message(
                    st.session_state.session_id,
                    "assistant",
                    full_response,
                    sources=sources,
                    selected_doc_ids=active_doc_ids, # type: ignore
                )
                if sources:
                    st.session_state["_latest_sources"] = sources
                    st.session_state["_latest_debug"] = None
                elif debug_logs:
                    st.session_state["_latest_debug"] = {
                        **debug_info,
                        "final_answer_len": len(full_response),
                        "note": "No source nodes returned by retriever/query engine.",
                    }
                    st.session_state["_latest_sources"] = None
                st.rerun()

        if st.session_state.get("_latest_sources"):
            with st.expander("Sources"):
                for src in st.session_state["_latest_sources"]:
                    preview = (src.get("text_preview") or "").replace("\n", " ").strip()
                    if len(preview) > 100:
                        preview = preview[:100] + "..."
                    st.markdown(f"- `{src['filename']}` | score `{src['score']:.2f}` | {preview}")
        elif debug_logs and st.session_state.get("_latest_debug"):
            with st.expander("Debug"):
                st.json(st.session_state["_latest_debug"])
