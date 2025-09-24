# rag_system/config.py

import os
import warnings
from dotenv import load_dotenv

# Disable all telemetry sources
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACES"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()

# Determine which provider to use
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

llm = None
embeddings = None

if USE_OLLAMA:
    from langchain.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        # LLM for reasoning
        llm = Ollama(
            model="qwen2.5:0.5b",
            base_url=ollama_base_url,
            temperature=0.1
        )
        
        # Embeddings for RAG
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
        
        print("✅ Using Ollama for LLM and Embeddings.")
        
    except Exception as e:
        print(f"❌ Error initializing Ollama: {e}. Falling back to Google Gen AI.")
        USE_OLLAMA = False

if not USE_OLLAMA:
    from crewai import LLM
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.1,
        api_key=google_api_key,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
        timeout=60,
        max_retries=3,
        wait_time=5,
    )
    print("✅ Using Google Gen AI for LLM and Embeddings.")

# Export both LLM and Embeddings from a single config file
__all__ = ["llm", "embeddings"]