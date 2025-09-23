# main.py

import os
import streamlit as st
import warnings
import atexit
from typing import Optional, List
import json
import re
import spacy
from datetime import datetime

from rag_system.crew_setup import (
    create_rag_crew,
    initialize_vector_store,
    verify_google_genai_setup,
    get_vector_store_info,
    reset_vector_store,
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
                return f"🤖 **Google Gen AI Enhanced Summary**: {str(enhanced_summary)}"

            except Exception as e:
                st.warning(f"Google Gen enhanced summary failed: {e}")
                return summary

        return summary

    except Exception as e:
        st.error(f"Google Gen auto summary generation failed: {e}")
        return None


# --- Streamlit UI setup ---
st.set_page_config(
    page_title="RAG Chat Assistant - Powered by Google Gen AI", layout="wide"
)
st.title("📚 RAG Chat Assistant")
st.caption("🤖 Powered by Google Gen AI Embeddings & Language Models")

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

# Sidebar for document upload and chat management
with st.sidebar:
    st.header("📁 Document Management")
    st.caption("🤖 Using Google Gen AI Embeddings")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or text files - processed with Google Gen AI",
    )

    # URL input
    urls_input = st.text_area(
        "Add URLs (one per line)",
        placeholder="https://example.com\nhttps://another-site.com",
        help="Add web content - embedded with Google Gen AI",
    )

    # Process documents button
    if st.button(
        "🔄 Process with Google Gen AI",
        type="primary",
        disabled=st.session_state.processing_docs,
    ):
        if uploaded_files or urls_input:
            st.session_state.processing_docs = True

            with st.spinner("Processing documents with Google Gen AI..."):
                try:
                    # Clear existing vector store first
                    if st.session_state.vector_store:
                        st.session_state.vector_store = None

                    documents = load_and_process_data(uploaded_files, urls_input)

                    if documents:
                        # Initialize new vector store with unique directory
                        st.session_state.vector_store = initialize_vector_store(
                            documents
                        )
                        st.session_state.documents_loaded = True

                        st.success(
                            f"✅ Processed {len(documents)} documents with Google Gen AI!"
                        )

                        # Generate Google Gen-powered summary
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

    # Vector store status
    if st.session_state.vector_store:
        store_info = get_vector_store_info(st.session_state.vector_store)
        st.success(f"✅ {store_info}")
    else:
        st.info("📝 Upload documents to start Google Gen-powered chat")

    st.divider()

    # Chat management
    st.header("💬 Chat Management")

    if st.button("➕ New Chat"):
        st.session_state.chat_id = create_new_chat()
        st.session_state.messages = []
        st.rerun()

    # Cleanup controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clean Chats", help="Remove chats with no messages"):
            deleted_count = cleanup_empty_chats()
            if deleted_count > 0:
                st.success(f"✅ Cleaned {deleted_count} empty chats")
            else:
                st.info("ℹ️ No empty chats found")
            st.rerun()

    with col2:
        if st.button("🗑️ Reset Vector Store", help="Clear all vector data"):
            reset_vector_store()
            st.session_state.vector_store = None
            st.session_state.documents_loaded = False
            st.rerun()

    # Load existing chats with proper error handling
    try:
        existing_chats = get_all_chats()
        if existing_chats:
            st.subheader(f"Previous Chats ({len(existing_chats)})")

            for chat in existing_chats[-5:]:
                col1, col2 = st.columns([3, 1])

                with col1:
                    chat_title = chat.get("title", "New Chat")[:30]
                    chat_id = chat.get("id")

                    if chat_id and st.button(
                        f"💬 {chat_title}...", key=f"load_{chat_id}"
                    ):
                        st.session_state.chat_id = chat_id
                        st.session_state.messages = load_chat_history(chat_id)
                        st.rerun()

                with col2:
                    if chat_id and st.button(
                        "🗑️", key=f"delete_{chat_id}", help="Delete chat"
                    ):
                        delete_chat(chat_id)
                        st.rerun()
        else:
            st.info("No previous chats found")

    except Exception as e:
        st.error(f"Error loading chats: {e}")

# Main chat interface
st.header("💭 Google Gen AI Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("type") == "mistral_summary":
            st.caption("🤖 Auto-generated by Google Gen AI")

# Query input
if prompt := st.chat_input(
    "Ask me anything about your documents (powered by Google Gen AI)...",
    disabled=not st.session_state.vector_store,
):
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and process documents with Google Gen AI first")
    else:
        if not st.session_state.chat_id:
            st.session_state.chat_id = create_new_chat()

        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        save_chat_message(st.session_state.chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Google Gen AI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

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
                    else:
                        response = f"🤖 **Google Gen AI Response**: {response}"

                    message_placeholder.markdown(response)

                    # Save assistant response
                    assistant_message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(assistant_message)
                    save_chat_message(st.session_state.chat_id, "assistant", response)

                except Exception as e:
                    error_msg = f"❌ Google Gen AI processing error: {str(e)}"
                    message_placeholder.error(error_msg)

                    error_message = {"role": "assistant", "content": error_msg}
                    st.session_state.messages.append(error_message)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔧 System Info"):
        with st.expander("Google Gen AI System Information", expanded=True):
            setup_ok, setup_message = verify_google_genai_setup()
            st.write(f"**Google Gen API**: {'✅ Active' if setup_ok else '❌ Issue'}")
            st.write(f"**Details**: {setup_message}")
            st.write(f"**spaCy**: {'✅ Loaded' if nlp_spacy else '❌ Not loaded'}")
            st.write(
                f"**Vector Store**: {'✅ Ready' if st.session_state.vector_store else '❌ Not initialized'}"
            )
            st.write(f"**Total Chats**: {get_chat_count()}")
            if st.session_state.vector_store:
                st.write(
                    f"**Store Info**: {get_vector_store_info(st.session_state.vector_store)}"
                )

with col2:
    st.session_state.show_query_analysis = st.checkbox(
        "🔍 Show Query Analysis", value=False
    )

with col3:
    if st.button("🔄 Complete Reset"):
        reset_vector_store()
        st.session_state.vector_store = None
        st.session_state.documents_loaded = False
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()

# Custom CSS
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
    
    .mistral-badge {
        background-color: #ff7000;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)
