# rag_system/config.py

import os
import warnings
from crewai import LLM
from dotenv import load_dotenv

# Disable all telemetry sources
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACES"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()

# Centralized LLM setup
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.1,
    api_key=google_api_key,
)