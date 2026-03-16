import os
import uuid
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_MESSAGES_TABLE = os.getenv("SUPABASE_MESSAGES_TABLE", "legal_chat_messages")

if SUPABASE_URL:
    SUPABASE_URL = SUPABASE_URL.strip().strip('"').strip("'")
if SUPABASE_KEY:
    SUPABASE_KEY = SUPABASE_KEY.strip().strip('"').strip("'")

class SupabaseChatDB:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            self.client = None # type: ignore
            print("Warning: Supabase credentials missing. Persistence disabled.")
        else:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.messages_table = SUPABASE_MESSAGES_TABLE

    def is_connected(self) -> bool:
        return self.client is not None

    def create_session(self) -> str:
        """Creates a new session and returns the session_id (UUID string)."""
        if not self.is_connected():
            return str(uuid.uuid4())
        
        session_id = str(uuid.uuid4())
        try:
            self.client.table("sessions").insert({"id": session_id}).execute()
        except Exception as e:
            print(f"Error creating session in Supabase: {e}")
            # Fallback to local-only if DB write fails
        return session_id

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        selected_doc_ids: Optional[List[str]] = None,
    ):
        """Persists a chat message associated with a session."""
        if not self.is_connected():
            return

        try:
            data = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources": sources if sources else [],
                "selected_doc_ids": selected_doc_ids if selected_doc_ids else [],
            }
            self.client.table(self.messages_table).insert(data).execute()
        except Exception as e:
            print(f"Error saving message to Supabase: {e}")

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a specific session sorted by time."""
        if not self.is_connected():
            return []

        try:
            response = self.client.table(self.messages_table) \
                .select("*") \
                .eq("session_id", session_id) \
                .order("created_at") \
                .execute()
            return response.data # type: ignore
        except Exception as e:
            print(f"Error fetching messages from Supabase: {e}")
            return []

    def create_document(
        self,
        doc_id: str,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        session_id: Optional[str] = None,
    ) -> None:
        if not self.is_connected():
            return
        try:
            payload: Dict[str, Any] = {
                "id": doc_id,
                "filename": filename,
                "file_path": file_path,
                "file_size_bytes": file_size_bytes,
                "ingest_status": "processing",
            }
            if session_id:
                payload["session_id"] = session_id
            self.client.table("documents").insert(payload).execute()
        except Exception as e:
            print(f"Error creating document row in Supabase: {e}")

    def update_document_status(
        self,
        doc_id: str,
        ingest_status: str,
        chunk_count: int = 0,
        ingest_error: Optional[str] = None,
    ) -> None:
        if not self.is_connected():
            return
        try:
            payload: Dict[str, Any] = {
                "ingest_status": ingest_status,
                "chunk_count": chunk_count,
            }
            if ingest_error:
                payload["ingest_error"] = ingest_error
            self.client.table("documents").update(payload).eq("id", doc_id).execute()
        except Exception as e:
            print(f"Error updating document status in Supabase: {e}")

    def list_documents(self) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        try:
            response = (
                self.client.table("documents")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            rows = response.data or []  # type: ignore
            return [row for row in rows if int(row.get("is_deleted") or 0) == 0]
        except Exception as e:
            print(f"Error listing documents from Supabase: {e}")
            return []

    def soft_delete_document(self, doc_id: str) -> bool:
        if not self.is_connected():
            return False
        try:
            self.client.table("documents").update({"is_deleted": 1}).eq("id", doc_id).execute()
            return True
        except Exception as e:
            print(f"Error soft deleting document in Supabase: {e}")
            return False

    def delete_pdf_from_storage(self, bucket: str, file_path: str) -> bool:
        if not self.is_connected():
            return False
        try:
            self.client.storage.from_(bucket).remove([file_path])
            return True
        except Exception as e:
            print(f"Error deleting PDF from Supabase Storage: {e}")
            return False

    def download_pdf_from_storage(self, bucket: str, file_path: str) -> Optional[bytes]:
        if not self.is_connected():
            return None
        try:
            return self.client.storage.from_(bucket).download(file_path)
        except Exception as e:
            print(f"Error downloading PDF from Supabase Storage: {e}")
            return None

    def upload_pdf_to_storage(self, bucket: str, file_path: str, content: bytes) -> bool:
        if not self.is_connected():
            return False
        try:
            self.client.storage.from_(bucket).upload(
                path=file_path,
                file=content,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )
            return True
        except Exception as e:
            print(f"Error uploading PDF to Supabase Storage: {e}")
            return False

# Initialize a global instance for convenience
supabase_db = SupabaseChatDB()
