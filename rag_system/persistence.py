# rag_system/persistence.py

import sqlite3
import uuid

DB_PATH = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    conn.commit()
    conn.close()


def create_new_chat():
    init_db()
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (id) VALUES (?)", (chat_id,))
    conn.commit()
    conn.close()
    return chat_id


def get_all_chats():
    """Returns list of tuples (chat_id, title) for compatibility with selectbox"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get chats with their first message as title
    cursor.execute(
        """
    SELECT DISTINCT c.id, 
           COALESCE(m.content, 'New Chat') as title
    FROM chats c
    LEFT JOIN (
        SELECT chat_id, content, 
               ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY timestamp) as rn
        FROM messages 
        WHERE role = 'user'
    ) m ON c.id = m.chat_id AND m.rn = 1
    ORDER BY c.created_at DESC
    """
    )

    # Return as list of tuples (id, title)
    chats = cursor.fetchall()
    conn.close()

    return chats if chats else []


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
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
