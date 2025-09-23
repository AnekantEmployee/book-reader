# rag_system/agents.py

from crewai import Agent
from rag_system.config import llm


class RagAgents:
    def __init__(self, tools):
        self.tools = tools

    def qna_agent(self):
        return Agent(
            role="RAG Assistant",
            goal="Answer user questions concisely and accurately using retrieved documents. Focus on providing direct, fact-based answers. Do not handle any formatting.",
            backstory="""You are an intelligent RAG assistant that understands context and provides helpful, concise answers based on available information. Your goal is to be accurate and direct, citing information when it is available in the retrieved documents.""",
            llm=llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=90,
            memory=False,
        )

    def summarization_agent(self):
        return Agent(
            role="Document Summarizer",
            goal="Create comprehensive and concise document summaries (150-250 words maximum).",
            backstory="""You are an expert at creating clear, concise summaries. You extract key information and present it in an organized, readable format for easy understanding.""",
            llm=llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=60,
            memory=False,
        )

    def formatting_agent(self):
        return Agent(
            role="Answer Formatter",
            goal="Take a raw text answer and re-format it into a structured, easy-to-read response using Markdown, including bullet points and headings.",
            backstory="""You are an expert at taking raw information and making it presentable. Your only job is to format a final answer. You must follow the user's explicit formatting rules, such as using bullet points and headings to break up content.""",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=30,
            memory=False,
        )