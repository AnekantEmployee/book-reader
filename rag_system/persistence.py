# rag_system/persistence.py
import sqlite3
import uuid

DB_PATH = "chat_history.db"

def init_db():
    """Initializes the SQLite database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            audio_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)
    conn.commit()
    conn.close()

def create_new_chat():
    """Creates a new chat entry in the database and returns its ID."""
    init_db()
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (id) VALUES (?)", (chat_id,))
    conn.commit()
    conn.close()
    return chat_id

def get_all_chats():
    """Retrieves all chat IDs and their first message for display in the sidebar."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, m.content
        FROM chats c
        LEFT JOIN (
            SELECT chat_id, content FROM messages GROUP BY chat_id
        ) m ON c.id = m.chat_id
        ORDER BY c.created_at DESC
    """)
    chats = [{"id": row[0], "title": row[1] if row[1] else "New Chat"} for row in cursor.fetchall()]
    conn.close()
    return chats

def load_chat_history(chat_id):
    """Loads all messages for a given chat ID from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, audio_path FROM messages WHERE chat_id = ? ORDER BY timestamp", (chat_id,))
    history = []
    for row in cursor.fetchall():
        msg = {"role": row[0], "content": row[1]}
        if row[2]:
            msg['audio'] = row[2]
        history.append(msg)
    conn.close()
    return history

def save_chat_message(chat_id, role, content, audio_path=None):
    """Saves a single message to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, role, content, audio_path) VALUES (?, ?, ?, ?)",
                   (chat_id, role, content, audio_path))
    conn.commit()
    conn.close()