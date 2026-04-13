import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
chat_collection = client["dochat_db"]["chat_sessions"]


def save_message(session_id: str, role: str, content: str):
    """Append a single message to the session's chat history in MongoDB."""
    chat_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow(),
                }
            },
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )


def load_messages(session_id: str) -> list:
    """Load chat history for a session. Returns list of {role, content} dicts."""
    doc = chat_collection.find_one({"session_id": session_id})
    if doc:
        return doc.get("messages", [])
    return []


def save_uploaded_doc(session_id: str, filename: str):
    """Track which documents were uploaded in this session."""
    chat_collection.update_one(
        {"session_id": session_id},
        {
            "$addToSet": {"uploaded_docs": filename},
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )


def load_uploaded_docs(session_id: str) -> list:
    """Load list of uploaded documents for a session."""
    doc = chat_collection.find_one({"session_id": session_id})
    if doc:
        return doc.get("uploaded_docs", [])
    return []


def clear_session_history(session_id: str):
    """Delete chat history and uploaded docs record for a session."""
    chat_collection.delete_one({"session_id": session_id})
