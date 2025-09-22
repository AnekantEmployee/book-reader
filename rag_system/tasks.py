# rag_system/tasks.py
from crewai import Task
from textwrap import dedent

class RagTasks:
    def __init__(self, qna_tool):
        self.qna_tool = qna_tool

    def answer_question_task(self, agent, question):
        return Task(
            description=dedent(f"""
                Retrieve information and answer the following question:
                
                Question: '{question}'
                
                You MUST use the 'RAG Retrieval Tool' to find relevant information before
                you formulate your answer. Your final response must be based strictly on the
                information you retrieved from the tool. Do not add any external knowledge.
                """),
            expected_output=dedent("""
                A clear, concise, and accurate answer to the question,
                strictly based on the retrieved context. If the context does not
                contain the answer, state that the information is not available.
                """),
            agent=agent,
            tools=[self.qna_tool]
        )

    def summarize_documents_task(self, agent, documents):
        return Task(
            description=dedent(f"""
                Read and summarize the following documents into a comprehensive overview.
                
                Documents: {documents}
                
                Your summary should be structured and easy to read, capturing all key points.
                """),
            expected_output=dedent("""
                A well-structured summary of the documents provided.
                """),
            agent=agent
        )