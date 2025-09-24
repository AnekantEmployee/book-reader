# rag_system/persistence.py - Updated with chat-specific document storage

import sqlite3
import uuid
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

DB_PATH = "chat_history.db"
CHAT_VECTOR_STORES_PATH = "./chat_vector_stores"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            has_documents INTEGER DEFAULT 0,
            document_count INTEGER DEFAULT 0,
            vector_store_path TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            audio_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
        """
    )

    # New table for chat-specific document metadata
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            document_name TEXT,
            document_type TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            chunk_count INTEGER DEFAULT 0,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    conn.close()


def create_new_chat():
    init_db()
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, "New Chat"))
    conn.commit()
    conn.close()

    # Create dedicated vector store directory for this chat
    chat_vector_path = Path(CHAT_VECTOR_STORES_PATH) / chat_id
    chat_vector_path.mkdir(parents=True, exist_ok=True)

    return chat_id


def get_all_chats():
    """Returns list of dictionaries with chat info including document status"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT c.id,
               COALESCE(c.title, COALESCE(m.content, 'New Chat')) as title,
               c.created_at,
               c.has_documents,
               c.document_count,
               COUNT(DISTINCT cd.id) as actual_doc_count
        FROM chats c
        LEFT JOIN (
            SELECT chat_id, content,
                   ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY timestamp) as rn
            FROM messages
            WHERE role = 'user'
        ) m ON c.id = m.chat_id AND m.rn = 1
        LEFT JOIN chat_documents cd ON c.id = cd.chat_id
        GROUP BY c.id, c.title, c.created_at, c.has_documents, c.document_count
        ORDER BY c.created_at DESC
        """
    )

    chats = []
    for row in cursor.fetchall():
        chat = {
            "id": row[0],
            "title": row[1][:50] if row[1] else "New Chat",
            "created_at": row[2],
            "has_documents": bool(row[3]),
            "document_count": row[5] if row[5] else 0,  # Use actual count
        }
        chats.append(chat)

    conn.close()
    return chats if chats else []


def update_chat_document_info(chat_id, document_count=0, has_documents=False):
    """Update chat's document information"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Also update the vector store path
    vector_store_path = str(Path(CHAT_VECTOR_STORES_PATH) / chat_id)

    cursor.execute(
        """
        UPDATE chats 
        SET has_documents = ?, document_count = ?, vector_store_path = ?
        WHERE id = ?
        """,
        (1 if has_documents else 0, document_count, vector_store_path, chat_id),
    )
    conn.commit()
    conn.close()


def add_chat_document(chat_id, document_name, document_type, chunk_count=0):
    """Add document metadata for a specific chat"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_documents (chat_id, document_name, document_type, chunk_count)
        VALUES (?, ?, ?, ?)
        """,
        (chat_id, document_name, document_type, chunk_count),
    )
    conn.commit()
    conn.close()


def get_chat_documents(chat_id):
    """Get all documents for a specific chat"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT document_name, document_type, upload_timestamp, chunk_count FROM chat_documents WHERE chat_id = ? ORDER BY upload_timestamp",
        (chat_id,),
    )

    documents = []
    for row in cursor.fetchall():
        doc = {
            "name": row[0],
            "type": row[1],
            "uploaded_at": row[2],
            "chunk_count": row[3],
        }
        documents.append(doc)

    conn.close()
    return documents


def get_chat_vector_store_path(chat_id):
    """Get the vector store path for a specific chat"""
    return str(Path(CHAT_VECTOR_STORES_PATH) / chat_id)


def chat_has_vector_store(chat_id):
    """Check if a chat has an existing vector store"""
    vector_store_path = Path(CHAT_VECTOR_STORES_PATH) / chat_id
    return vector_store_path.exists() and any(vector_store_path.iterdir())


def delete_chat_vector_store(chat_id):
    """Delete the vector store for a specific chat"""
    try:
        vector_store_path = Path(CHAT_VECTOR_STORES_PATH) / chat_id
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Error deleting vector store for chat {chat_id}: {e}")
        return False


def load_chat_history(chat_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, audio_path FROM messages WHERE chat_id = ? ORDER BY timestamp",
        (chat_id,),
    )

    history = []
    for row in cursor.fetchall():
        msg = {"role": row[0], "content": row[1]}
        if row[2]:
            msg["audio"] = row[2]
        history.append(msg)

    conn.close()
    return history


def save_chat_message(chat_id, role, content, audio_path=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, audio_path) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, audio_path),
    )
    conn.commit()
    conn.close()


def delete_chat(chat_id):
    """Delete chat and its associated vector store"""
    init_db()

    # Delete vector store first
    delete_chat_vector_store(chat_id)

    # Delete from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    cursor.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


def get_chat_count():
    """Get total number of chats"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chats")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_message_count(chat_id=None):
    """Get message count for specific chat or all chats"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if chat_id:
        cursor.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
    else:
        cursor.execute("SELECT COUNT(*) FROM messages")

    count = cursor.fetchone()[0]
    conn.close()
    return count


def cleanup_empty_chats():
    """Remove chats with no messages and clean up their vector stores"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # First get the chat IDs that will be deleted
    cursor.execute(
        """
        SELECT id FROM chats 
        WHERE id NOT IN (SELECT DISTINCT chat_id FROM messages WHERE chat_id IS NOT NULL)
        """
    )

    empty_chat_ids = [row[0] for row in cursor.fetchall()]

    # Delete their vector stores
    for chat_id in empty_chat_ids:
        delete_chat_vector_store(chat_id)

    # Delete from database
    cursor.execute(
        """
        DELETE FROM chats 
        WHERE id NOT IN (SELECT DISTINCT chat_id FROM messages WHERE chat_id IS NOT NULL)
        """
    )

    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count


def cleanup_orphaned_vector_stores():
    """Clean up vector store directories that don't have corresponding chats"""
    init_db()

    if not Path(CHAT_VECTOR_STORES_PATH).exists():
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chats")
    valid_chat_ids = {row[0] for row in cursor.fetchall()}
    conn.close()

    cleaned_count = 0
    for vector_dir in Path(CHAT_VECTOR_STORES_PATH).iterdir():
        if vector_dir.is_dir() and vector_dir.name not in valid_chat_ids:
            try:
                shutil.rmtree(vector_dir, ignore_errors=True)
                cleaned_count += 1
            except Exception as e:
                print(f"Error cleaning orphaned vector store {vector_dir}: {e}")

    return cleaned_count


def get_chat_info(chat_id):
    """Get detailed information about a specific chat"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT c.title, c.created_at, c.has_documents, c.document_count,
               COUNT(DISTINCT m.id) as message_count,
               COUNT(DISTINCT cd.id) as actual_doc_count
        FROM chats c
        LEFT JOIN messages m ON c.id = m.chat_id
        LEFT JOIN chat_documents cd ON c.id = cd.chat_id
        WHERE c.id = ?
        GROUP BY c.id, c.title, c.created_at, c.has_documents, c.document_count
        """,
        (chat_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "id": chat_id,
            "title": row[0],
            "created_at": row[1],
            "has_documents": bool(row[2]),
            "document_count": row[5] if row[5] else 0,  # Use actual count
            "message_count": row[4] if row[4] else 0,
            "vector_store_exists": chat_has_vector_store(chat_id),
        }
    return None
