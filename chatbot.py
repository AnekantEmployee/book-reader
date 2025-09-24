# Updated chatbot.py with chat-specific document context management

import os
import streamlit as st
import warnings
import atexit
from typing import List
import json
import re
import spacy
from datetime import datetime
from pathlib import Path

from rag_system.crew_setup import (
    create_rag_crew,
    initialize_chat_vector_store,
    load_chat_vector_store,
    add_documents_to_chat_store,
    reset_chat_vector_store,
    get_chat_vector_store_info,
    verify_google_genai_setup,
    cleanup_on_exit,
)
from rag_system.persistence import (
    save_chat_message,
    load_chat_history,
    create_new_chat,
    get_all_chats,
    delete_chat,
    cleanup_empty_chats,
    get_chat_count,
    get_chat_documents,
    get_chat_info,
    cleanup_orphaned_vector_stores,
    chat_has_vector_store,
)
from rag_system.data_loaders import load_and_process_data
from rag_system.text_to_speech import generate_audio
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from dotenv import load_dotenv

# Register cleanup function
atexit.register(cleanup_on_exit)

# Disable all telemetry and traces
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACES"] = "false"
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()


# Initialize NLP components
@st.cache_resource
def initialize_nlp_components():
    """Initialize spaCy for query preprocessing"""
    nlp_spacy = None

    try:
        nlp_spacy = spacy.load("en_core_web_sm")
        st.success("✅ spaCy model loaded successfully!")
    except OSError:
        st.warning("⚠️ spaCy English model not found. Installing...")
        try:
            os.system("python -m spacy download en_core_web_sm")
            nlp_spacy = spacy.load("en_core_web_sm")
            st.success("✅ spaCy model installed successfully!")
        except Exception as e:
            st.error(f"❌ Could not install spaCy model: {e}")
            nlp_spacy = None

    return nlp_spacy


class GoogleGenAIQueryPreprocessor:
    """Query preprocessing optimized for Google Gen AI"""

    def __init__(self, nlp_spacy):
        self.nlp_spacy = nlp_spacy

    def preprocess_query(self, query: str) -> dict:
        """Preprocess query for optimal Google Gen AI performance"""
        result = {
            "original": query,
            "cleaned": query.strip(),
            "entities": [],
            "keywords": [],
            "semantic_variants": [],
            "intent": "general",
            "mistral_optimized": query,
        }

        try:
            # Basic cleaning
            result["cleaned"] = re.sub(r"\s+", " ", query.strip().lower())

            # SpaCy processing if available
            if self.nlp_spacy:
                doc = self.nlp_spacy(query)

                # Extract named entities
                result["entities"] = [(ent.text, ent.label_) for ent in doc.ents]

                # Extract keywords optimized for Google Gen
                result["keywords"] = [
                    token.lemma_
                    for token in doc
                    if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]
                    and not token.is_stop
                    and len(token.text) > 2
                ]

                # Determine intent for Google Gen optimization
                if any(token.lemma_ in ["what", "define", "explain"] for token in doc):
                    result["intent"] = "definition"
                elif any(token.lemma_ in ["how", "process", "step"] for token in doc):
                    result["intent"] = "process"
                elif any(
                    token.lemma_ in ["compare", "difference", "versus"] for token in doc
                ):
                    result["intent"] = "comparison"
                elif any(token.lemma_ in ["list", "show", "all"] for token in doc):
                    result["intent"] = "listing"

            # Optimize query for Google Gen's semantic understanding
            result["mistral_optimized"] = self._optimize_for_mistral(
                query, result["keywords"]
            )

        except Exception as e:
            st.warning(f"Query preprocessing error: {e}")

        return result

    def _optimize_for_mistral(self, query: str, keywords: List[str]) -> str:
        """Optimize query specifically for Google Gen AI's semantic search"""
        try:
            # Keep original query as primary, enhance with keywords if needed
            optimized = query.strip()

            # If query is very short, enhance with keywords
            if len(optimized.split()) < 3 and keywords:
                key_terms = " ".join(keywords[:3])
                optimized = f"{optimized} {key_terms}".strip()

            return optimized

        except Exception:
            return query


def create_mistral_summary(documents, nlp_spacy=None) -> str:
    """Create document summary optimized for Google Gen AI processing"""
    try:
        if not documents:
            return "No documents to summarize."

        # Combine document content
        full_content = ""
        for doc in documents:
            if hasattr(doc, "page_content"):
                full_content += doc.page_content + "\n\n"
            elif isinstance(doc, str):
                full_content += doc + "\n\n"

        if not full_content.strip():
            return "Documents appear to be empty."

        # Limit content for optimal Google Gen processing
        if len(full_content) > 8000:
            full_content = full_content[:8000] + "..."

        # Enhanced summarization with spaCy if available
        if nlp_spacy:
            try:
                doc = nlp_spacy(full_content)
                sentences = [sent.text for sent in doc.sents]

                if len(sentences) > 8:
                    # Select key sentences for Google Gen
                    summary_sentences = (
                        sentences[:2]
                        + sentences[len(sentences) // 2 : len(sentences) // 2 + 2]
                        + sentences[-2:]
                    )
                    summary = " ".join(summary_sentences)
                else:
                    summary = " ".join(sentences[:4])

                return f"📄 **Document Summary** (Powered by Google Gen AI): {summary}"

            except Exception as e:
                st.warning(f"Enhanced summarization failed: {e}")

        # Fallback summarization
        sentences = full_content.split(". ")
        if len(sentences) > 4:
            summary = ". ".join(sentences[:3]) + "..."
        else:
            summary = (
                full_content[:400] + "..." if len(full_content) > 400 else full_content
            )

        return f"📄 **Document Summary** (Powered by Google Gen AI): {summary}"

    except Exception as e:
        st.error(f"Error creating Google Gen summary: {e}")
        return "❌ Could not generate document summary with Google Gen AI."


def generate_mistral_auto_summary(documents, vector_store, nlp_spacy=None):
    """Generate automatic summary using Google Gen AI"""
    try:
        if not documents:
            return None

        # Create basic summary
        summary = create_mistral_summary(documents, nlp_spacy)

        # Enhance with Google Gen RAG if vector store available
        if vector_store:
            try:
                from rag_system.tools import EnhancedRagRetrievalTool
                from rag_system.agents import RagAgents
                from rag_system.tasks import RagTasks

                qna_tool = EnhancedRagRetrievalTool(vector_store=vector_store)
                agents = RagAgents([qna_tool])
                tasks = RagTasks(qna_tool)

                summary_agent = agents.qna_agent()
                summary_task = tasks.summarize_documents_task(summary_agent, documents)

                summary_crew = Crew(
                    agents=[summary_agent],
                    tasks=[summary_task],
                    process=Process.sequential,
                    verbose=False,
                    memory=False,
                    cache=False,
                )

                enhanced_summary = summary_crew.kickoff()
                return str(enhanced_summary)

            except Exception as e:
                st.warning(f"Google Gen enhanced summary failed: {e}")
                return summary

        return summary

    except Exception as e:
        st.error(f"Google Gen auto summary generation failed: {e}")
        return None

def generate_subtopic_suggestions(documents, vector_store):
    """Generate subtopic suggestions using a dedicated CrewAI agent."""
    try:
        if not documents:
            return None

        from rag_system.tools import EnhancedRagRetrievalTool
        from rag_system.agents import RagAgents
        from rag_system.tasks import RagTasks

        qna_tool = EnhancedRagRetrievalTool(vector_store=vector_store)
        agents = RagAgents([qna_tool])
        tasks = RagTasks(qna_tool)
        
        # New agent and task for subtopic extraction
        analyzer_agent = agents.document_analyzer_agent()
        subtopic_task = tasks.extract_subtopics_task(analyzer_agent, documents)

        analyzer_crew = Crew(
            agents=[analyzer_agent],
            tasks=[subtopic_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            cache=False,
        )

        subtopic_list = analyzer_crew.kickoff()
        return str(subtopic_list)

    except Exception as e:
        st.warning(f"Failed to generate subtopic suggestions: {e}")
        return None


def switch_to_chat(chat_id):
    """Switch to a specific chat and load its context - FIXED VERSION"""
    try:
        st.session_state.chat_id = chat_id
        st.session_state.messages = load_chat_history(chat_id)

        vector_store = load_chat_vector_store(chat_id)

        if vector_store:
            try:
                test_results = vector_store.similarity_search("test", k=1)
                # FIX 1: Use consistent naming
                st.session_state.vector_store = (
                    vector_store  # Changed from 'vectorstore'
                )
                st.session_state.documents_loaded = True

                chat_docs = get_chat_documents(chat_id)
                if chat_docs:
                    st.success(
                        f"Switched to chat with {len(chat_docs)} documents - Vector store loaded successfully"
                    )
            except Exception as e:
                st.warning(f"Vector store exists but not functional: {e}")
                st.session_state.vector_store = None  # Changed from 'vectorstore'
                st.session_state.documents_loaded = False
        else:
            st.session_state.vector_store = None  # Changed from 'vectorstore'
            st.session_state.documents_loaded = False

    except Exception as e:
        st.error(f"Error switching to chat: {e}")
        st.session_state.vector_store = None  # Changed from 'vectorstore'
        st.session_state.documents_loaded = False


# --- Streamlit UI setup ---
st.set_page_config(
    page_title="RAG Chat Assistant - Powered by Google Gen AI", layout="wide"
)
st.title("📚 RAG Chat Assistant")
st.caption("🤖 Powered by Google Gen AI Embeddings & Language Models")
st.caption("💬 Each chat maintains its own document context")

# Verify Google Gen setup at startup
setup_ok, setup_message = verify_google_genai_setup()
if setup_ok:
    st.success(f"✅ {setup_message}")
else:
    st.error(f"❌ Google Gen AI Setup Issue: {setup_message}")
    st.stop()

# Initialize NLP components
nlp_spacy = initialize_nlp_components()
query_processor = GoogleGenAIQueryPreprocessor(nlp_spacy)

# Initialize session state with unique keys
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "processing_docs" not in st.session_state:
    st.session_state.processing_docs = False
if "current_chat_docs" not in st.session_state:
    st.session_state.current_chat_docs = []

# Sidebar for document upload and chat management
with st.sidebar:
    st.header("💬 Chat Management")

    # Current chat info
    if st.session_state.chat_id:
        chat_info = get_chat_info(st.session_state.chat_id)
        if chat_info:
            st.info(f"📍 **Current Chat**")
            st.caption(f"ID: {st.session_state.chat_id[:8]}...")
            st.caption(f"Documents: {chat_info.get('document_count', 0)}")
            st.caption(f"Messages: {chat_info.get('message_count', 0)}")
    else:
        st.info("No active chat selected")

    # New chat button
    if st.button("➕ New Chat", type="primary"):
        st.session_state.chat_id = create_new_chat()
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.documents_loaded = False
        st.session_state.current_chat_docs = []
        st.rerun()

    st.divider()

    # Chat history with enhanced display
    st.header("📝 Previous Chats")

    # Cleanup controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clean Empty", help="Remove chats with no messages"):
            deleted_count = cleanup_empty_chats()
            if deleted_count > 0:
                st.success(f"✅ Cleaned {deleted_count} empty chats")
            else:
                st.info("ℹ️ No empty chats found")
            st.rerun()

    with col2:
        if st.button("🗂️ Clean Orphans", help="Clean orphaned vector stores"):
            cleaned_count = cleanup_orphaned_vector_stores()
            if cleaned_count > 0:
                st.success(f"✅ Cleaned {cleaned_count} orphaned stores")
            else:
                st.info("ℹ️ No orphaned stores found")
            st.rerun()

    # Load existing chats with enhanced info
    try:
        existing_chats = get_all_chats()
        if existing_chats:
            st.caption(f"Found {len(existing_chats)} chats")

            for chat in existing_chats[-8:]:  # Show more chats
                chat_title = chat.get("title", "New Chat")[:25]
                chat_id = chat.get("id")
                doc_count = chat.get("document_count", 0)
                has_docs = chat.get("has_documents", False)

                # Create visual indicators
                doc_icon = "📄" if has_docs else "💬"
                doc_info = f" ({doc_count} docs)" if doc_count > 0 else ""

                # Current chat indicator
                current_indicator = "👉 " if chat_id == st.session_state.chat_id else ""

                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button(
                        f"{current_indicator}{doc_icon} {chat_title}{doc_info}",
                        key=f"load_{chat_id}",
                        help=f"Switch to this chat",
                    ):
                        switch_to_chat(chat_id)
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        delete_chat(chat_id)
                        if chat_id == st.session_state.chat_id:
                            st.session_state.chat_id = None
                            st.session_state.messages = []
                            st.session_state.vector_store = None
                            st.session_state.documents_loaded = False
                        st.rerun()
        else:
            st.info("No previous chats found")

    except Exception as e:
        st.error(f"Error loading chats: {e}")

    st.divider()

    # Document Management Section
    st.header("📄 Document Management")
    st.caption(f"🤖 Chat-specific document context")

    # Show current chat's documents
    if st.session_state.chat_id:
        current_docs = get_chat_documents(st.session_state.chat_id)
        if current_docs:
            st.success(f"📊 **Current Chat Documents ({len(current_docs)})**")
            for i, doc in enumerate(current_docs[:5]):  # Show first 5
                doc_name = doc.get("name", "Unknown")
                doc_type = doc.get("type", "unknown")
                st.caption(f"{i+1}. {doc_name} ({doc_type})")

            if len(current_docs) > 5:
                st.caption(f"... and {len(current_docs) - 5} more")

            # Option to reset current chat's documents
            if st.button(
                "🗑️ Clear Chat Documents", help="Remove all documents from current chat"
            ):
                if st.session_state.chat_id:
                    reset_chat_vector_store(st.session_state.chat_id)
                    st.session_state.vector_store = None
                    st.session_state.documents_loaded = False
                    st.success("✅ Chat documents cleared")
                    st.rerun()
        else:
            st.info("📝 No documents in current chat")
    else:
        st.warning("⚠️ Select or create a chat first")

    # File upload (only if chat is selected)
    if st.session_state.chat_id:
        uploaded_files = st.file_uploader(
            "Upload Documents to Current Chat",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or text files - processed with Google Gen AI",
            key="chat_file_upload",
        )

        # URL input
        urls_input = st.text_area(
            "Add URLs to Current Chat (one per line)",
            placeholder="https://example.com\nhttps://another-site.com",
            help="Add web content - embedded with Google Gen AI",
            key="chat_url_input",
        )

        # Process documents button
        process_button_text = (
            "➕ Add to Current Chat"
            if st.session_state.vector_store
            else "📄 Create Documents for Chat"
        )

        if st.button(process_button_text, type="primary", disabled=st.session_state.processing_docs):
            if uploaded_files or urls_input:
                st.session_state.processing_docs = True

                with st.spinner("Processing documents with Google Gen AI..."):
                    try:
                        documents = load_and_process_data(uploaded_files, urls_input)

                        if documents:
                            if st.session_state.vector_store:
                                st.session_state.vector_store = add_documents_to_chat_store(
                                    st.session_state.chat_id,
                                    st.session_state.vector_store,
                                    documents
                                )
                            else:
                                st.session_state.vector_store = initialize_chat_vector_store(
                                    st.session_state.chat_id, documents
                                )

                            st.session_state.documents_loaded = True

                            # Generate Google Gen-powered summary for new documents
                            with st.spinner("Generating Google Gen AI summary..."):
                                auto_summary = generate_mistral_auto_summary(
                                    documents, st.session_state.vector_store, nlp_spacy
                                )

                                if auto_summary:
                                    summary_message = {
                                        "role": "assistant",
                                        "content": auto_summary,
                                        "timestamp": datetime.now().isoformat(),
                                        "type": "mistral_summary",
                                    }
                                    st.session_state.messages.append(summary_message)

                                    if st.session_state.chat_id:
                                        save_chat_message(
                                            st.session_state.chat_id,
                                            "assistant",
                                            auto_summary,
                                        )

                            with st.spinner("Generating suggested subtopics..."):
                                subtopics = generate_subtopic_suggestions(documents, st.session_state.vector_store)
                                if subtopics:
                                    subtopic_message = {"role": "assistant", "content": f"Here are some questions you might want to ask:\n\n{subtopics}", "type": "subtopics"}
                                    st.session_state.messages.append(subtopic_message)
                                    save_chat_message(st.session_state.chat_id, "assistant", f"Suggested Questions:\n{subtopics}")
                        else:
                            st.error("❌ No valid documents found to process")

                    except Exception as e:
                        st.error(f"❌ Error processing with Google Gen AI: {e}")
                        st.exception(e)
                    finally:
                        st.session_state.processing_docs = False
                        st.rerun()
            else:
                st.warning("⚠️ Please upload files or add URLs first")
    else:
        st.info("📝 Create or select a chat first to add documents")

# Main chat interface
st.header("💭 Google Gen AI Chat Interface")

# Enhanced status information
if st.session_state.chat_id:
    col1, col2, col3 = st.columns(3)
    with col1:
        chat_status = "Active" if st.session_state.chat_id else "None"
        st.metric(
            "Current Chat", "Active" if st.session_state.chat_id else "None", "💬"
        )
    with col2:
        doc_status = "Active" if st.session_state.vector_store else "Empty"
        doc_count = (
            len(get_chat_documents(st.session_state.chat_id))
            if st.session_state.chat_id
            else 0
        )
        st.metric("Documents", doc_count, "📄")
    with col3:
        vector_status = "Loaded" if st.session_state.vector_store else "None"
        st.metric("Vector Store", vector_status, "🔊")
else:
    st.info("👆 Please create or select a chat from the sidebar to begin")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("type") == "mistral_summary":
            st.caption("🤖 Auto-generated by Google Gen AI")

# Enhanced query input with better UX
if st.session_state.chat_id:
    if st.session_state.vector_store:
        placeholder_text = (
            "Ask me anything about your documents (powered by Google Gen AI)..."
        )
        input_disabled = False
    else:
        placeholder_text = "Upload documents first, or ask general questions..."
        input_disabled = False
else:
    placeholder_text = "Please create or select a chat first..."
    input_disabled = True

if prompt := st.chat_input(placeholder_text, disabled=input_disabled):
    if not st.session_state.chat_id:
        st.warning("⚠️ Please create or select a chat first")
    else:
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        save_chat_message(st.session_state.chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Google Gen AI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if st.session_state.vector_store:
                with st.spinner("🤖 Processing with Google Gen AI..."):
                    try:
                        # Preprocess query for Google Gen
                        query_info = query_processor.preprocess_query(prompt)

                        # Use Google Gen-optimized query
                        mistral_query = query_info["mistral_optimized"]

                        # Create and run Google Gen RAG crew
                        rag_crew = create_rag_crew(
                            mistral_query, st.session_state.vector_store
                        )
                        result = rag_crew.kickoff()

                        # Format response
                        response = str(result)

                        # Add Google Gen branding to response
                        if query_info["intent"] != "general":
                            response = f"*[{query_info['intent'].title()} Query - Google Gen AI]*\n\n{response}"

                        message_placeholder.markdown(response)

                        # Save assistant response
                        assistant_message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(assistant_message)
                        save_chat_message(
                            st.session_state.chat_id, "assistant", response
                        )

                    except Exception as e:
                        error_msg = f"❌ Google Gen AI processing error: {str(e)}"
                        message_placeholder.error(error_msg)

                        error_message = {"role": "assistant", "content": error_msg}
                        st.session_state.messages.append(error_message)
            else:
                # General response without document context
                response = f"I don't have any documents loaded for this chat yet. Please upload some documents first to enable document-based Q&A, or I can help with general questions.\n\nYour question: '{prompt}'"
                message_placeholder.markdown(response)

                assistant_message = {"role": "assistant", "content": response}
                st.session_state.messages.append(assistant_message)
                save_chat_message(st.session_state.chat_id, "assistant", response)

# Enhanced footer with chat-specific info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.vector_store:
        st.success("✅ Chat documents active")
    else:
        st.info("📄 No documents in current chat")
with col2:
    st.caption("Powered by Google Gen AI")
with col3:
    chat_id = st.session_state.get("chat_id", "None")
    if chat_id and chat_id != "None":
        st.caption(f"Chat: {chat_id[:8]}...")
    else:
        st.caption("Chat: None")

# Custom CSS for chat-specific styling
st.markdown(
    """
<style>
    .stChat > div > div > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #ff7000;
    }
    
    .chat-specific-indicator {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .mistral-badge {
        background-color: #ff7000;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 5px;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    
    .current-chat-indicator {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 5px;
        border-radius: 3px;
    }
</style>
""",
    unsafe_allow_html=True,
)
