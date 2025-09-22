# rag_system/agents.py
from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_community.llms import MistralAI
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
llm = MistralAI(mistral_api_key=mistral_api_key)

# Initialize the search tool
serper_search_tool = SerperDevTool()

class RagAgents:
    def __init__(self, tools):
        self.tools = tools

    def qna_agent(self):
        return Agent(
            role='RAG Assistant',
            goal='Answer user questions based ONLY on the provided context. If the answer is not in the context, state that you cannot answer.',
            backstory="You are a specialized RAG (Retrieval-Augmented Generation) assistant. Your core directive is to be an expert in the given documents. You strictly adhere to the information provided and will never use external knowledge to formulate your answer.",
            llm=llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )

    def summarization_agent(self):
        return Agent(
            role='Document Summarizer',
            goal='Create a concise and accurate summary of the provided documents.',
            backstory="You are a meticulous summarization expert. Your job is to read long documents and distill their essence into a clear, easy-to-understand summary. Your summaries must be factual and capture all main points.",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )