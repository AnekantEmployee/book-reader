# agents.py - Updated agent definitions

from crewai import Agent
from rag_system.config import llm


class RagAgents:
    def __init__(self, tools):
        self.tools = tools

    def qna_agent(self):
        return Agent(
            role="RAG Assistant",
            goal="Answer user questions accurately using retrieved documents. Always provide helpful information when relevant context is available, even if not perfectly matching the query.",
            backstory="""You are an intelligent RAG assistant that analyzes retrieved context carefully. 
            You should:
            1. Always examine the retrieved context thoroughly
            2. Provide helpful information even if it's related but not exactly matching the query
            3. Clearly state when information is partial or when you need to infer connections
            4. Never say 'no information found' when there is actually relevant context available
            5. If the query asks about concept A vs B, but context has A vs C, explain what you know about A and mention the limitation""",
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
