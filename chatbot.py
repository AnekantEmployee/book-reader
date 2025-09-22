# main.py
import streamlit as st
from rag_system.crew_setup import create_rag_crew, initialize_vector_store
from rag_system.persistence import save_chat_message, load_chat_history, create_new_chat, get_all_chats
from rag_system.data_loaders import load_and_process_data
from rag_system.text_to_speech import generate_audio
import os

# Set up environment variables
from dotenv import load_dotenv
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("📚 RAG Chat Assistant")

# Initialize session state for the chat
if 'chat_id' not in st.session_state:
    st.session_state['chat_id'] = create_new_chat()
    st.session_state['messages'] = []
    
# --- Sidebar for chat management ---
with st.sidebar:
    st.header("Chat Management")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state['chat_id'] = create_new_chat()
        st.session_state['messages'] = []
        st.success("New chat created!")
        st.rerun()

    st.markdown("---")
    st.header("Chat History")
    all_chats = get_all_chats()
    for chat in all_chats:
        if st.button(f"📄 {chat['title']}", key=chat['id'], use_container_width=True):
            st.session_state['chat_id'] = chat['id']
            st.session_state['messages'] = load_chat_history(chat['id'])
            st.rerun()

# --- Main chat interface ---
# Display existing messages from the session state
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])
    if 'audio' in msg and os.path.exists(msg['audio']):
        st.audio(msg['audio'])

# File uploader section
with st.expander("Upload Documents & URLs"):
    uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
    urls_input = st.text_area("Enter URLs (websites, Google Docs, YouTube) separated by new lines")
    
    if st.button("Process Documents"):
        if uploaded_files or urls_input:
            with st.spinner("Processing documents... This may take a moment."):
                docs = load_and_process_data(uploaded_files, urls_input)
                # Initialize the persistent vector store
                initialize_vector_store(docs)
                st.success("Documents processed and ready for Q&A!")
        else:
            st.warning("Please upload a file or enter URLs.")

# Chat input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to history and display
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save user message to the database
    save_chat_message(st.session_state['chat_id'], "user", prompt)

    with st.spinner("Thinking..."):
        try:
            # Check if vector store is initialized
            if 'vector_store' not in st.session_state or st.session_state['vector_store'] is None:
                st.error("Please upload and process documents first to enable the RAG system.")
                st.stop()
            
            # Create and kickoff the crew
            crew = create_rag_crew(prompt)
            result = crew.kickoff()
            
            # Add assistant response to history and display
            st.session_state['messages'].append({"role": "assistant", "content": result})
            with st.chat_message("assistant"):
                st.write(result)
                
            # Generate and display audio overview
            audio_file = generate_audio(result)
            st.session_state['messages'][-1]['audio'] = audio_file
            st.audio(audio_file)

            # Save assistant message to the database
            save_chat_message(st.session_state['chat_id'], "assistant", result, audio_file)
            
        except ValueError as e:
            st.error(f"Error: {e}")