# main.py
import streamlit as st
from rag_system.crew_setup import create_rag_crew, initialize_vector_store
from rag_system.persistence import save_chat_message, load_chat_history, create_new_chat, get_all_chats, delete_chat
from rag_system.data_loaders import load_and_process_data
from rag_system.text_to_speech import generate_audio
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks

import os
from dotenv import load_dotenv
load_dotenv()

# --- Streamlit UI setup ---
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("📚 RAG Chat Assistant")

# Initialize session state variables
if 'chat_id' not in st.session_state:
    st.session_state['chat_id'] = create_new_chat()
    st.session_state['messages'] = []
    st.session_state['expander_state'] = True
    st.session_state['vector_store'] = None

# --- Sidebar for chat management ---
def handle_delete_chat():
    delete_chat(st.session_state['chat_id'])
    st.session_state['messages'] = []

with st.sidebar:
    st.header("Chat Management")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state['chat_id'] = create_new_chat()
            st.session_state['messages'] = []
            st.rerun()
    with col2:
        if st.button("🗑️ Delete Chat", use_container_width=True, on_click=handle_delete_chat):
            pass # The action is handled by the on_click callback

    st.markdown("---")
    st.header("Chat History")
    all_chats = get_all_chats()
    for chat in all_chats:
        if st.button(f"📄 {chat['title']}", key=chat['id'], use_container_width=True):
            st.session_state['chat_id'] = chat['id']
            st.session_state['messages'] = load_chat_history(chat['id'])
            st.rerun()

# --- Main chat interface ---
# Display messages from session state
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])
    if 'audio' in msg and os.path.exists(msg['audio']):
        st.audio(msg['audio'])

# File uploader section
with st.expander("Upload Documents & URLs", expanded=st.session_state['expander_state']):
    uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
    urls_input = st.text_area("Enter URLs (websites, Google Docs, YouTube) separated by new lines")
    
    if st.button("Process Documents"):
        if uploaded_files or urls_input:
            with st.spinner("Processing documents... This may take a moment."):
                docs = load_and_process_data(uploaded_files, urls_input)

            # 📝 New: Check if documents were successfully loaded
            if not docs:
                st.error("No documents were loaded. Please check your files or URLs.")
                st.session_state['expander_state'] = True
                st.stop()
                
            st.session_state['vector_store'] = initialize_vector_store(docs)
            st.session_state['expander_state'] = False

            st.success("Documents processed and ready for Q&A!")
            st.write("### Generating a summary of the content...")
            
            try:
                agents = RagAgents([])
                tasks = RagTasks(None)
                
                summarization_agent = agents.summarization_agent()
                summarize_task = tasks.summarize_documents_task(summarization_agent, docs)

                summarization_crew = Crew(
                    agents=[summarization_agent],
                    tasks=[summarize_task],
                    process=Process.sequential,
                    verbose=True
                )
                summary_result = summarization_crew.kickoff()
                
                # 📝 Fix: Display the summary outside of any chat message container
                st.info(summary_result)
                
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
        else:
            st.warning("Please upload a file or enter URLs.")

# Chat input
if prompt := st.chat_input("Ask a question about the documents..."):
    # User message in a chat message box
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    save_chat_message(st.session_state['chat_id'], "user", prompt)

    with st.spinner("Thinking..."):
        try:
            vector_store = st.session_state['vector_store']
            if vector_store is None:
                st.error("Please upload and process documents first to enable the RAG system.")
                st.stop()
            
            crew = create_rag_crew(prompt, vector_store)
            crew_output = crew.kickoff()
            
            # 📝 Fix: Extract the raw string content from the CrewOutput object
            result_text = crew_output.raw
            
            # Assistant response in a separate chat message box
            st.session_state['messages'].append({"role": "assistant", "content": result_text})
            with st.chat_message("assistant"):
                st.write(result_text)
                
            audio_file = generate_audio(result_text)
            st.session_state['messages'][-1]['audio'] = audio_file
            st.audio(audio_file)
            
            # 📝 Fix: Pass the raw string to the save function
            save_chat_message(st.session_state['chat_id'], "assistant", result_text, audio_file)
            
        except ValueError as e:
            st.error(f"Error: {e}")