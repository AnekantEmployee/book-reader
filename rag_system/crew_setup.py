# Enhanced crew_setup.py with chat-specific vector store management

import os
import warnings
import chromadb
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from rag_system.tools import EnhancedRagRetrievalTool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
import time
from rag_system.config import llm

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


def initialize_mistral_embeddings():
    """Initialize Google Gen AI embeddings with robust error handling"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError(
                "Error: GOOGLE_API_KEY not found in environment variables."
            )

        # Initialize Google Gen embeddings with extended timeouts
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
            timeout=60,
            max_retries=3,
            wait_time=5,
        )

        # Test the embeddings with timeout handling
        try:
            with st.spinner("Testing Google Gen AI connection..."):
                test_embedding = embeddings.embed_query("test initialization")
                if test_embedding and len(test_embedding) > 0:
                    return embeddings
                else:
                    raise RuntimeError("Google Gen embeddings returned empty result")
        except Exception as e:
            if "Server disconnected" in str(e):
                st.error(
                    "Google Gen AI server connection lost. Please check your internet connection and try again."
                )
                raise RuntimeError(f"Google Gen AI connection error: {e}")
            else:
                raise RuntimeError(f"Google Gen embeddings test failed: {e}")

    except Exception as e:
        st.error(f"Failed to initialize Google Gen AI embeddings: {e}")
        raise


def load_chat_vector_store(chat_id):
    """Debug version of load_chat_vector_store to identify issues"""
    print(f"DEBUG: Loading vector store for chat {chat_id}")

    try:
        if not chat_has_vector_store(chat_id):
            print(f"DEBUG: No vector store found for chat {chat_id}")
            return None

        vector_store_path = get_chat_vector_store_path(chat_id)
        print(f"DEBUG: Vector store path: {vector_store_path}")

        if not Path(vector_store_path).exists():
            print(f"DEBUG: Vector store path does not exist: {vector_store_path}")
            return None

        # Initialize embeddings
        embeddings = initialize_mistral_embeddings()
        print("DEBUG: Embeddings initialized successfully")

        # Try to load the existing vector store
        try:
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings,
                collection_name=f"chat_{chat_id}_documents",
            )
            print("DEBUG: Chroma vector store created")

            # Test if it's working
            test_results = vector_store.similarity_search("test", k=1)
            print(f"DEBUG: Test search returned {len(test_results)} results")

            # Store client reference
            _chat_chroma_clients[chat_id] = vector_store

            # Get document info
            chat_documents = get_chat_documents(chat_id)
            doc_count = len(chat_documents)
            print(f"DEBUG: Found {doc_count} documents in database")

            return vector_store

        except Exception as e:
            print(f"DEBUG: Error loading vector store: {e}")
            return None

    except Exception as e:
        print(f"DEBUG: Error in load_chat_vector_store: {e}")
        return None


def initialize_chat_vector_store(chat_id, documents):
    """Initialize vector store for a specific chat"""
    global _chat_chroma_clients

    try:
        if not documents:
            raise ValueError("No documents provided for vector store initialization")

        if not chat_id:
            raise ValueError("Chat ID is required for chat-specific vector store")

        # Clean up any existing clients for this chat
        if chat_id in _chat_chroma_clients:
            try:
                client = _chat_chroma_clients[chat_id]
                if hasattr(client, "_client") and client._client:
                    client._client.stop()
                del _chat_chroma_clients[chat_id]
            except Exception:
                pass

        # Initialize Google Gen AI embeddings
        embeddings = initialize_mistral_embeddings()

        # Get chat-specific directory
        persist_directory = get_chat_vector_store_path(chat_id)

        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Create new vector store with persistent storage
        collection_name = f"chat_{chat_id}_documents"

        try:
            st.info(f"Creating vector store for chat at: {persist_directory}")

            # Create vector store with persistent storage
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )

            # Store client reference
            _chat_chroma_clients[chat_id] = vector_store

            # Update chat document information
            update_chat_document_info(chat_id, len(documents), True)

            # Save document metadata
            for i, doc in enumerate(documents):
                doc_name = f"Document_{i+1}"
                doc_type = "unknown"

                # Try to extract document info from metadata
                if hasattr(doc, "metadata") and doc.metadata:
                    doc_name = doc.metadata.get("source", doc_name)
                    if "pdf" in doc_name.lower():
                        doc_type = "pdf"
                    elif "txt" in doc_name.lower():
                        doc_type = "txt"
                    elif "http" in doc_name.lower():
                        doc_type = "url"

                add_chat_document(chat_id, doc_name, doc_type, 1)

            st.success(
                f"Created chat-specific vector store with {len(documents)} documents"
            )

            return vector_store

        except Exception as e:
            st.error(f"Chat vector store creation failed: {e}")
            # Try in-memory as final fallback
            try:
                st.info("Creating in-memory vector store as fallback")
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=f"memory_{chat_id}",
                )
                st.warning("Using in-memory vector store (will not persist)")
                return vector_store
            except Exception as final_e:
                st.error(f"Even in-memory store failed: {final_e}")
                raise RuntimeError(
                    f"Complete vector store initialization failure: {final_e}"
                )

    except Exception as e:
        st.error(f"Chat vector store initialization failed: {e}")
        raise


def add_documents_to_chat_store(chat_id, vector_store, new_documents):
    """Add new documents to existing chat-specific vector store"""
    try:
        # Add the new documents
        vector_store.add_documents(new_documents)

        # Update chat metadata
        existing_docs = get_chat_documents(chat_id)
        new_count = len(existing_docs) + len(new_documents)
        update_chat_document_info(chat_id, new_count, True)

        # Add new document metadata
        start_index = len(existing_docs)
        for i, doc in enumerate(new_documents):
            doc_name = f"Document_{start_index + i + 1}"
            doc_type = "unknown"

            # Try to extract document info from metadata
            if hasattr(doc, "metadata") and doc.metadata:
                doc_name = doc.metadata.get("source", doc_name)
                if "pdf" in doc_name.lower():
                    doc_type = "pdf"
                elif "txt" in doc_name.lower():
                    doc_type = "txt"
                elif "http" in doc_name.lower():
                    doc_type = "url"

            add_chat_document(chat_id, doc_name, doc_type, 1)

        st.success(f"Added {len(new_documents)} documents to chat (Total: {new_count})")
        return vector_store

    except Exception as e:
        st.error(f"Failed to add documents to chat store: {e}")
        # Fallback to recreating
        return initialize_chat_vector_store(chat_id, new_documents)


def reset_chat_vector_store(chat_id):
    """Reset/clear the vector store for a specific chat"""
    global _chat_chroma_clients

    try:
        # Clean up client first
        if chat_id in _chat_chroma_clients:
            try:
                client = _chat_chroma_clients[chat_id]
                if hasattr(client, "_client") and client._client:
                    client._client.stop()
                del _chat_chroma_clients[chat_id]
            except Exception:
                pass

        # Remove persistent directory
        vector_store_path = get_chat_vector_store_path(chat_id)
        if Path(vector_store_path).exists():
            shutil.rmtree(vector_store_path, ignore_errors=True)

        # Update chat metadata
        update_chat_document_info(chat_id, 0, False)

        # Clear document metadata
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

        try:
            # Get count from database metadata
            chat_documents = get_chat_documents(chat_id)
            if chat_documents:
                count = len(chat_documents)
                return f"Chat vector store contains {count} documents"
            else:
                # Fallback to collection count
                collection = vector_store._collection
                count = collection.count()
                return f"Chat vector store contains {count} document chunks"
        except Exception:
            return "Chat vector store is active but details unavailable"

    except Exception as e:
        return f"Error getting chat vector store info: {e}"


def create_rag_crew(user_query, vector_store):
    """Create RAG crew with enhanced error handling"""
    try:
        if vector_store is None:
            raise ValueError("Vector store is not initialized.")

        # Validate vector store functionality
        try:
            test_search = vector_store.similarity_search("test", k=1)
        except Exception as e:
            raise ValueError(f"Vector store is not functional: {e}")

        # Create the enhanced tool
        qna_tool = EnhancedRagRetrievalTool(vector_store=vector_store)

        # Initialize agents and tasks
        agents = RagAgents([qna_tool])
        tasks = RagTasks(qna_tool)

        # Create a two-step sequential crew
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


def verify_google_genai_setup():
    """Verify Google Gen AI setup with connection testing"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return False, "GOOGLE_API_KEY not found in environment variables."

        # Initialize a test LLM instance
        from rag_system.config import llm as global_llm

        # Test embedding generation
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
        )
        test_embedding = embeddings.embed_query("test")

        if not test_embedding or len(test_embedding) == 0:
            return False, "Google Gen embeddings returned empty result."

        # Test the LLM by running a simple Crew
        try:
            from crewai import Agent, Task, Crew, Process

            test_agent = Agent(
                role="Test Agent",
                goal="Validate LLM connectivity by responding to a simple prompt.",
                backstory="An agent designed to check the LLM connection.",
                llm=global_llm,
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

            # Use kickoff() to test the LLM
            test_response = test_crew.kickoff()

            if test_response:
                return (
                    True,
                    f"Google Gen AI verified (dimension: {len(test_embedding)})",
                )
            else:
                return False, "LLM failed to generate a response via a test crew."

        except Exception as e:
            error_msg = str(e)
            if "AuthenticationError" in error_msg:
                return False, "Authentication failed. The API key is not valid."
            elif "Server disconnected" in error_msg:
                return False, "Google Gen AI server connection lost."
            elif "timeout" in error_msg.lower():
                return False, "Google Gen AI request timed out."
            else:
                return False, f"Google Gen AI error: {e}."

    except Exception as e:
        return False, f"Google Gen AI setup error: {e}."


# Cleanup function for app shutdown
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
    "verify_google_genai_setup",
    "cleanup_on_exit",
]
