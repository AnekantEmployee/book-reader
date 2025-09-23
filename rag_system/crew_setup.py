# rag_system/crew_setup.py

import os
import warnings
import chromadb
import uuid
import shutil
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

# Global variable to track Chroma clients
_chroma_clients = {}


def cleanup_chroma_clients():
    """Clean up existing Chroma clients to avoid instance conflicts"""
    global _chroma_clients
    try:
        for client_path, client in _chroma_clients.items():
            try:
                if hasattr(client, "_client") and client._client:
                    client._client.stop()
            except Exception:
                pass
        _chroma_clients.clear()

        # Force garbage collection
        import gc

        gc.collect()

    except Exception as e:
        print(f"Warning: Could not cleanup Chroma clients: {e}")


def reset_chroma_directory(persist_directory):
    """Safely reset Chroma directory"""
    try:
        if os.path.exists(persist_directory):
            # First try to cleanup any existing clients
            cleanup_chroma_clients()

            # Wait a moment for cleanup
            time.sleep(0.5)

            # Remove the directory
            shutil.rmtree(persist_directory, ignore_errors=True)

            # Wait for filesystem to catch up
            time.sleep(0.5)

            return True
    except Exception as e:
        st.warning(f"Could not reset Chroma directory: {e}")
        return False


def initialize_mistral_embeddings():
    """Initialize Google Gen AI embeddings with robust error handling"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("Error: GOOGLE_API_KEY not found in environment variables.")

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
                    st.success("✅ Google Gen AI embeddings initialized successfully")
                    return embeddings
                else:
                    raise RuntimeError("Google Gen embeddings returned empty result")
        except Exception as e:
            if "Server disconnected" in str(e):
                st.error(
                    "❌ Google Gen AI server connection lost. Please check your internet connection and try again."
                )
                raise RuntimeError(f"Google Gen AI connection error: {e}")
            else:
                raise RuntimeError(f"Google Gen embeddings test failed: {e}")

    except Exception as e:
        st.error(f"❌ Failed to initialize Google Gen AI embeddings: {e}")
        raise


def create_unique_persist_dir(base_dir="./chroma_db"):
    """Create a unique persist directory to avoid conflicts"""
    timestamp = str(int(time.time()))
    session_id = str(uuid.uuid4())[:8]
    unique_dir = f"{base_dir}_{timestamp}_{session_id}"
    return unique_dir


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
            verbose=True, # Set verbose to True for debugging
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


def initialize_vector_store(documents, persist_directory=None):
    """Initialize vector store with complete fix for instance conflicts"""
    global _chroma_clients

    try:
        if not documents:
            raise ValueError("No documents provided for vector store initialization")

        # Initialize Google Gen AI embeddings
        embeddings = initialize_mistral_embeddings()

        # Clean up any existing clients first
        cleanup_chroma_clients()

        # Create unique directory if none provided or if conflicts exist
        if persist_directory is None:
            persist_directory = create_unique_persist_dir()

        # Check if directory already has conflicts
        if os.path.exists(persist_directory):
            try:
                # Try to create a test client to check for conflicts
                test_client = chromadb.PersistentClient(path=persist_directory)
                test_client.heartbeat()
                test_collection = test_client.get_or_create_collection(
                    "test_collection"
                )

                # If successful, try to use existing store
                try:
                    existing_store = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=embeddings,
                        collection_name="default_collection",
                    )
                    test_results = existing_store.similarity_search("test", k=1)

                    # Store the client reference
                    _chroma_clients[persist_directory] = existing_store

                    st.info(
                        "📂 Successfully loaded existing Google Gen-powered vector store"
                    )
                    return existing_store

                except Exception as inner_e:
                    st.warning(f"Existing store incompatible: {inner_e}")
                    # Reset and create new
                    test_client.reset()

            except Exception as e:
                st.warning(f"Directory conflict detected: {e}")
                # Create new unique directory
                persist_directory = create_unique_persist_dir()

        # Create new vector store with unique collection name
        collection_name = f"documents_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        try:
            st.info(f"🔄 Creating new vector store at: {persist_directory}")

            # Method 1: Direct Chroma client approach
            try:
                chroma_client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=chromadb.config.Settings(
                        anonymized_telemetry=False, allow_reset=True
                    ),
                )

                # Create collection
                collection = chroma_client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )

                # Create vector store from client
                vector_store = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )

                # Add documents
                vector_store.add_documents(documents)

                # Store client reference
                _chroma_clients[persist_directory] = vector_store

                st.success(
                    f"✅ Created new Google Gen vector store with {len(documents)} documents"
                )
                return vector_store

            except Exception as e1:
                st.warning(f"Method 1 failed: {e1}")

                # Method 2: from_documents with unique directory
                persist_directory = create_unique_persist_dir()

                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )

                # Store client reference
                _chroma_clients[persist_directory] = vector_store

                st.success(
                    f"✅ Created Google Gen vector store (fallback method) with {len(documents)} documents"
                )
                return vector_store

        except Exception as e:
            st.error(f"❌ All vector store creation methods failed: {e}")

            # Final fallback: In-memory store
            try:
                st.info("🔄 Creating in-memory vector store as final fallback")

                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=f"memory_{int(time.time())}",
                )

                st.warning("⚠️ Using in-memory vector store (data will not persist)")
                return vector_store

            except Exception as final_e:
                st.error(f"❌ Even in-memory store failed: {final_e}")
                raise RuntimeError(
                    f"Complete vector store initialization failure: {final_e}"
                )

    except Exception as e:
        st.error(f"❌ Vector store initialization failed: {e}")
        raise


def reset_vector_store(persist_directory=None):
    """Reset/clear the vector store completely"""
    global _chroma_clients

    try:
        # Clean up clients first
        cleanup_chroma_clients()

        if persist_directory:
            if reset_chroma_directory(persist_directory):
                st.success(f"✅ Vector store cleared: {persist_directory}")
            else:
                st.warning(f"⚠️ Could not fully clear: {persist_directory}")
        else:
            # Clear all possible directories
            import glob

            chroma_dirs = glob.glob("./chroma_db*")
            cleared_count = 0

            for dir_path in chroma_dirs:
                if reset_chroma_directory(dir_path):
                    cleared_count += 1

            if cleared_count > 0:
                st.success(f"✅ Cleared {cleared_count} vector store directories")
            else:
                st.info("ℹ️ No vector stores to clear")

    except Exception as e:
        st.error(f"❌ Error during vector store reset: {e}")


def get_vector_store_info(vector_store):
    """Get information about the current vector store"""
    try:
        if not vector_store:
            return "No Google Gen vector store initialized"

        try:
            collection = vector_store._collection
            count = collection.count()
            collection_name = getattr(collection, "name", "unknown")
            return f"Google Gen vector store '{collection_name}' contains {count} document chunks"
        except Exception:
            return "Google Gen vector store is active but details unavailable"

    except Exception as e:
        return f"Error getting vector store info: {e}"


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
                return True, f"Google Gen AI verified (dimension: {len(test_embedding)})"
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