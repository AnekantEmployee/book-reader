# rag_system/tasks.py
from crewai import Task
from textwrap import dedent


class RagTasks:
    def __init__(self, qna_tool):
        self.qna_tool = qna_tool

    def answer_question_task(self, agent, question):
        return Task(
            description=dedent(
                f"""
            Answer the following question using the RAG Retrieval Tool:
            Question: '{question}'
            
            Use the tool to find relevant information and provide a concise answer (150-200 words).
            Be direct and informative. Do not mention using tools or context.
            """
            ),
            expected_output=dedent(
                """
            A clear, concise answer (150-200 words maximum) to the question,
            based on retrieved context. If no relevant information is found,
            state that clearly and briefly.
            """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def summarize_documents_task(self, agent, documents):
        return Task(
            description=dedent(
                f"""
            Create a concise summary of the uploaded documents.
            Focus on key points and main topics in 150-200 words maximum.
            Documents content: {str(documents)[:500]}...
            """
            ),
            expected_output=dedent(
                """
            A well-structured, concise summary (150-200 words) highlighting 
            the main topics and key information from the documents.
            """
            ),
            agent=agent,
        )
