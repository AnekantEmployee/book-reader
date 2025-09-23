# rag_system/data_loaders.py
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os

def load_and_process_data(uploaded_files, urls_input):
    documents = []

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the file temporarily
            file_path = Path(uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the document using the appropriate loader
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else: # assuming .txt
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(file_path))
            
            documents.extend(loader.load())
            os.remove(file_path) # Clean up the temporary file

    # Process URLs
    if urls_input:
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        for url in urls:
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = WebBaseLoader(url)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {url}: {e}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    return docs