# Enhanced crew_setup.py with chat-specific vector store management

import os
import warnings
import shutil
import json
from datetime import datetime
from pathlib import Path
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from rag_system.tools import EnhancedRagRetrievalTool
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
import time
# Import the centralized LLM and embeddings from config.py
from rag_system.config import llm, embeddings, USE_OLLAMA

# Import chat-specific persistence functions
from rag_system.persistence import (
    get_chat_vector_store_path,
    chat_has_vector_store,
    update_chat_document_info,
    add_chat_document,
    get_chat_documents,
)

# Disable all telemetry sources
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACES"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_SERVER_AUTHN_PROVIDER"] = ""

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()

# Global variables for chat-specific management
_chat_chroma_clients = {}  # Key: chat_id, Value: chroma client


def cleanup_chroma_clients():
    """Clean up existing Chroma clients to avoid instance conflicts"""
    global _chat_chroma_clients
    try:
        for chat_id, client in _chat_chroma_clients.items():
            try:
                if hasattr(client, "_client") and client._client:
                    client._client.stop()
            except Exception:
                pass
        _chat_chroma_clients.clear()

        # Force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not cleanup Chroma clients: {e}")


def verify_setup():
    """Verify LLM and embeddings setup with connection testing"""
    try:
        if USE_OLLAMA:
            from langchain.llms import Ollama
            from langchain_community.embeddings import OllamaEmbeddings
            test_llm = Ollama(model="qwen2.5:0.5b")
            test_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            from crewai import LLM
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                return False, "GOOGLE_API_KEY not found in environment variables."
            test_llm = LLM(model="gemini/gemini-2.5-flash", api_key=google_api_key)
            test_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_api_key)

        with st.spinner("Testing LLM and embeddings connection..."):
            test_embedding = test_embeddings.embed_query("test initialization")
            if not test_embedding:
                return False, "Embeddings returned empty result."

            from crewai import Agent, Task, Crew, Process
            test_agent = Agent(
                role="Test Agent",
                goal="Validate LLM connectivity.",
                backstory="An agent to check the LLM connection.",
                llm=test_llm,
                verbose=False,
                allow_delegation=False,
            )
            test_task = Task(
                description="Respond with a short confirmation message.",
                expected_output="A brief message confirming the task completion.",
                agent=test_agent,
            )
            test_crew = Crew(
                agents=[test_agent],
                tasks=[test_task],
                process=Process.sequential,
                verbose=False,
            )
            test_response = test_crew.kickoff()
            if test_response:
                model_name = "Ollama" if USE_OLLAMA else "Google Gen AI"
                return True, f"{model_name} verified (dimension: {len(test_embedding)})"
            else:
                return False, "LLM failed to generate a response via a test crew."
    except Exception as e:
        return False, f"Setup error: {e}"


def load_chat_vector_store(chat_id):
    """Load vector store for a specific chat"""
    try:
        if not chat_has_vector_store(chat_id):
            return None
        vector_store_path = get_chat_vector_store_path(chat_id)
        if not Path(vector_store_path).exists():
            return None

        # Use the configured embeddings
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings,
            collection_name=f"chat_{chat_id}_documents",
        )
        _chat_chroma_clients[chat_id] = vector_store
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def initialize_chat_vector_store(chat_id, documents):
    """Initialize vector store for a specific chat"""
    global _chat_chroma_clients

    try:
        if not documents:
            raise ValueError("No documents provided")
        if not chat_id:
            raise ValueError("Chat ID is required")

        if chat_id in _chat_chroma_clients:
            try:
                client = _chat_chroma_clients[chat_id]
                if hasattr(client, "_client") and client._client:
                    client._client.stop()
                del _chat_chroma_clients[chat_id]
            except Exception:
                pass

        persist_directory = get_chat_vector_store_path(chat_id)
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        collection_name = f"chat_{chat_id}_documents"

        st.info(f"Creating vector store for chat...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        _chat_chroma_clients[chat_id] = vector_store
        update_chat_document_info(chat_id, len(documents), True)
        for i, doc in enumerate(documents):
            doc_name = doc.metadata.get("source", f"Document_{i+1}")
            add_chat_document(chat_id, doc_name, "file", 1)
        st.success(
            f"Created chat-specific vector store with {len(documents)} documents"
        )
        return vector_store
    except Exception as e:
        st.error(f"Chat vector store initialization failed: {e}")
        raise


def add_documents_to_chat_store(chat_id, vector_store, new_documents):
    """Add new documents to existing chat-specific vector store"""
    try:
        vector_store.add_documents(new_documents)
        existing_docs = get_chat_documents(chat_id)
        new_count = len(existing_docs) + len(new_documents)
        update_chat_document_info(chat_id, new_count, True)
        for i, doc in enumerate(new_documents):
            doc_name = doc.metadata.get("source", f"Document_{len(existing_docs) + i + 1}")
            add_chat_document(chat_id, doc_name, "file", 1)
        st.success(f"Added {len(new_documents)} documents to chat (Total: {new_count})")
        return vector_store
    except Exception as e:
        st.error(f"Failed to add documents to chat store: {e}")
        return initialize_chat_vector_store(chat_id, new_documents)


def reset_chat_vector_store(chat_id):
    """Reset/clear the vector store for a specific chat"""
    global _chat_chroma_clients
    try:
        if chat_id in _chat_chroma_clients:
            client = _chat_chroma_clients[chat_id]
            if hasattr(client, "_client") and client._client:
                client._client.stop()
            del _chat_chroma_clients[chat_id]
        vector_store_path = get_chat_vector_store_path(chat_id)
        if Path(vector_store_path).exists():
            shutil.rmtree(vector_store_path, ignore_errors=True)
        update_chat_document_info(chat_id, 0, False)
        from rag_system.persistence import init_db
        init_db()
        import sqlite3
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
        conn.commit()
        conn.close()
        st.success(f"Reset vector store for chat")
    except Exception as e:
        st.error(f"Error during chat vector store reset: {e}")


def get_chat_vector_store_info(chat_id, vector_store):
    """Get information about the current chat's vector store"""
    try:
        if not vector_store:
            return "No vector store for this chat"
        chat_documents = get_chat_documents(chat_id)
        count = len(chat_documents)
        return f"Chat vector store contains {count} documents"
    except Exception as e:
        return f"Error getting chat vector store info: {e}"


def create_rag_crew(user_query, vector_store):
    """Create RAG crew with enhanced error handling"""
    try:
        if vector_store is None:
            raise ValueError("Vector store is not initialized.")
        qna_tool = EnhancedRagRetrievalTool(vector_store=vector_store)
        agents = RagAgents([qna_tool])
        tasks = RagTasks(qna_tool)
        rag_assistant = agents.qna_agent()
        answer_task = tasks.answer_question_task(rag_assistant, user_query)
        formatter_agent = agents.formatting_agent()
        format_task = tasks.format_answer_task(formatter_agent)
        rag_crew = Crew(
            agents=[rag_assistant, formatter_agent],
            tasks=[answer_task, format_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False,
            share_crew=False,
            full_output=True,
            step_callback=None,
            max_execution_time=120,
            max_retry_limit=2,
        )
        return rag_crew
    except Exception as e:
        st.error(f"Error creating RAG crew: {e}")
        raise


def cleanup_on_exit():
    """Cleanup function to call on app shutdown"""
    cleanup_chroma_clients()


# Export functions
__all__ = [
    "create_rag_crew",
    "initialize_chat_vector_store",
    "load_chat_vector_store",
    "add_documents_to_chat_store",
    "reset_chat_vector_store",
    "get_chat_vector_store_info",
    "verify_setup", # Changed function name to be more generic
    "cleanup_on_exit",
]