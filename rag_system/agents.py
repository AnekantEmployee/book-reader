# rag_system/agents.py
import os
from crewai import Agent
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# Disable telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Fix: Use correct model name format for Mistral
llm = ChatMistralAI(
    model="mistral-small-latest",  # Correct Mistral model name
    mistral_api_key=mistral_api_key,
    temperature=0.1,
)


class RagAgents:
    def __init__(self, tools):
        self.tools = tools

    def qna_agent(self):
        return Agent(
            role="RAG Assistant",
            goal="Answer user questions concisely (150-200 words) using retrieved documents. Understand semantic variations in queries.",
            backstory="""You are an intelligent RAG assistant that understands context and semantic variations. 
            When users ask about concepts using different spellings or formats (like "fav-up", "fav up", "favup"), 
            you understand they refer to the same thing. Provide concise, helpful answers based on available information.""",
            llm=llm,
            tools=self.tools,
            verbose=False,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=60,
            memory=False,
        )

    def summarization_agent(self):
        return Agent(
            role="Document Summarizer",
            goal="Create concise document summaries (150-200 words maximum).",
            backstory="""You are an expert at creating clear, concise summaries. 
            Extract key information and present it in an organized, readable format.""",
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=60,
            memory=False,
        )
