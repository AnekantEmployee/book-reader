# main.py
import os
import streamlit as st
from rag_system.crew_setup import create_rag_crew, initialize_vector_store
from rag_system.persistence import (
    save_chat_message,
    load_chat_history,
    create_new_chat,
    get_all_chats,
    delete_chat,
)
from rag_system.data_loaders import load_and_process_data
from rag_system.text_to_speech import generate_audio
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from dotenv import load_dotenv
import warnings

# Disable all telemetry and traces
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACES"] = "false"
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"

# Suppress warnings
warnings.filterwarnings("ignore")

load_dotenv()

# --- Streamlit UI setup ---
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("📚 RAG Chat Assistant")

# Initialize session state variables
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = create_new_chat()
    st.session_state["messages"] = []
    st.session_state["expander_state"] = True
    st.session_state["vector_store"] = None


# --- Sidebar for chat management ---
def handle_delete_chat():
    delete_chat(st.session_state["chat_id"])
    st.session_state["chat_id"] = create_new_chat()
    st.session_state["messages"] = []
    st.session_state["vector_store"] = None
    st.rerun()


def handle_new_chat():
    st.session_state["chat_id"] = create_new_chat()
    st.session_state["messages"] = []
    st.session_state["vector_store"] = None
    st.rerun()


with st.sidebar:
    st.subheader("💬 Chat Management")

    # Get all chats
    all_chats = get_all_chats()

    if all_chats:
        # Create options and format function for selectbox
        chat_options = [chat[0] for chat in all_chats]  # chat_id list

        def format_chat_title(chat_id):
            # Find the title for this chat_id
            for chat in all_chats:
                if chat[0] == chat_id:
                    title = chat[1] or f"Chat {chat_id[:8]}..."
                    return title[:50] + "..." if len(title) > 50 else title
            return f"Chat {chat_id[:8]}..."

        # Current chat selection
        if st.session_state["chat_id"] not in chat_options:
            st.session_state["chat_id"] = chat_options[0]

        selected_chat = st.selectbox(
            "Select a chat:",
            options=chat_options,
            format_func=format_chat_title,
            index=(
                chat_options.index(st.session_state["chat_id"])
                if st.session_state["chat_id"] in chat_options
                else 0
            ),
            key="chat_selector",
        )

        # Handle chat selection change
        if selected_chat != st.session_state["chat_id"]:
            st.session_state["chat_id"] = selected_chat
            st.session_state["messages"] = load_chat_history(selected_chat)
            st.rerun()

    # Chat management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New", key="new_chat_btn"):
            handle_new_chat()

    with col2:
        if st.button("🗑️ Delete", key="delete_chat_btn"):
            if len(all_chats) > 1:  # Don't delete if it's the only chat
                handle_delete_chat()
            else:
                st.warning("Cannot delete the only remaining chat.")

# --- Document upload and processing ---
with st.expander(
    "Upload Documents & URLs", expanded=st.session_state.get("expander_state", True)
):
    uploaded_files = st.file_uploader(
        "Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True
    )

    urls_input = st.text_area(
        "Enter URLs (websites, Google Docs, YouTube) separated by new lines",
        height=100,
        placeholder="https://github.com/topics/shodan-api\nhttps://example.com/document",
    )

    if st.button("Process Documents", type="primary"):
        if uploaded_files or urls_input:
            with st.spinner("Processing documents..."):
                try:
                    docs = load_and_process_data(uploaded_files, urls_input)

                    if not docs:
                        st.error(
                            "No documents were loaded. Please check your files or URLs."
                        )
                        st.stop()

                    st.session_state["vector_store"] = initialize_vector_store(docs)
                    st.session_state["expander_state"] = False
                    st.success(
                        f"✅ Documents processed! Loaded {len(docs)} document chunks."
                    )

                    # Generate summary with better error handling
                    try:
                        agents = RagAgents([])
                        tasks = RagTasks(None)

                        summarization_agent = agents.summarization_agent()
                        summarize_task = tasks.summarize_documents_task(
                            summarization_agent, docs
                        )

                        summarization_crew = Crew(
                            agents=[summarization_agent],
                            tasks=[summarize_task],
                            process=Process.sequential,
                            verbose=False,
                            memory=False,
                            share_crew=False,
                            full_output=False,
                        )

                        # Suppress output during execution
                        import sys
                        from io import StringIO

                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = StringIO()
                        sys.stderr = StringIO()

                        try:
                            summary_result = summarization_crew.kickoff()
                            summary_text = (
                                summary_result.raw
                                if hasattr(summary_result, "raw")
                                else str(summary_result)
                            )
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr

                        # Clean summary
                        if "Final Answer:" in summary_text:
                            summary_text = summary_text.split("Final Answer:")[
                                -1
                            ].strip()

                        summary_message = f"📋 **Document Summary:**\n\n{summary_text}"
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": summary_message}
                        )
                        save_chat_message(
                            st.session_state["chat_id"], "assistant", summary_message
                        )
                        st.rerun()

                    except Exception as summary_error:
                        # If summary fails, still show success for document processing
                        st.info(
                            "Documents processed successfully. Summary generation skipped due to configuration."
                        )

                except Exception as e:
                    st.error(f"❌ Failed to process documents: {str(e)}")
                    st.error("Please check your MISTRAL_API_KEY environment variable.")
        else:
            st.warning("⚠️ Please upload files or enter URLs first.")

# --- Chat interface ---
# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("audio"):
            st.audio(message["audio"])

# --- Chat input ---
if prompt := st.chat_input(
    "Ask a question about the documents...",
    disabled=st.session_state["vector_store"] is None,
):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    save_chat_message(st.session_state["chat_id"], "user", prompt)

    # Generate response
    with st.spinner("🤔 Thinking..."):
        try:
            vector_store = st.session_state["vector_store"]
            if vector_store is None:
                st.error("Please upload and process documents first.")
                st.stop()

            # Suppress output during crew execution
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            try:
                crew = create_rag_crew(prompt, vector_store)
                crew_output = crew.kickoff()
                result_text = (
                    crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Clean response
            if "Final Answer:" in result_text:
                result_text = result_text.split("Final Answer:")[-1].strip()

            # Remove trace-related content
            lines = result_text.split("\n")
            clean_lines = [
                line
                for line in lines
                if not any(
                    keyword in line.lower()
                    for keyword in [
                        "execution traces",
                        "view insights",
                        "agent decision",
                        "task execution",
                        "tool usage",
                        "detailed execution",
                    ]
                )
            ]
            result_text = "\n".join(clean_lines).strip()

            # Display response
            st.session_state["messages"].append(
                {"role": "assistant", "content": result_text}
            )
            with st.chat_message("assistant"):
                st.write(result_text)

            # Generate audio with better error handling
            try:
                audio_file = generate_audio(result_text)
                if audio_file:
                    st.session_state["messages"][-1]["audio"] = audio_file
                    st.audio(audio_file)
                    save_chat_message(
                        st.session_state["chat_id"],
                        "assistant",
                        result_text,
                        audio_file,
                    )
                else:
                    save_chat_message(
                        st.session_state["chat_id"], "assistant", result_text
                    )
            except Exception as audio_error:
                # Save without audio if TTS fails
                save_chat_message(st.session_state["chat_id"], "assistant", result_text)

        except Exception as e:
            error_message = f"❌ An error occurred: {str(e)}"
            st.error(error_message)
            st.error("Please check your MISTRAL_API_KEY and try again.")

# Show helpful message when no documents are loaded
if st.session_state["vector_store"] is None:
    st.info("📁 Upload documents above to start chatting with your RAG assistant!")
