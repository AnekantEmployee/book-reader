# rag_system/tasks.py

from crewai import Task
from textwrap import dedent
from typing import List, Any


class RagTasks:
    def __init__(self, qna_tool):
        self.qna_tool = qna_tool

    def answer_question_task(self, agent, question):
        """Enhanced question answering task with better prompts"""
        return Task(
            description=dedent(
                f"""
                Answer the following question using the Enhanced RAG Retrieval Tool with improved semantic search:
                
                Question: '{question}'
                
                Instructions:
                1. Use the tool to find the most relevant information
                2. Provide a comprehensive answer (200-400 words)
                3. Structure the response with clear main points
                4. Include specific details when available
                5. If information is limited, acknowledge this clearly
                6. Focus on being helpful and accurate
                7. Use natural, conversational language
                
                The tool will automatically handle query optimization and semantic search.
                """
            ),
            expected_output=dedent(
                """
                A well-structured, comprehensive answer (200-400 words) that:
                - Directly addresses the question
                - Uses information retrieved from the knowledge base
                - Is organized with clear main points
                - Includes relevant details and examples
                - Acknowledges any limitations in available information
                - Uses natural, helpful language
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def summarize_documents_task(self, agent, documents):
        """Enhanced document summarization task"""
        # Prepare document content for summarization
        doc_content = ""
        if documents:
            for i, doc in enumerate(documents[:5]):  # Limit to first 5 docs
                if hasattr(doc, "page_content"):
                    content = doc.page_content[:1000]  # Limit content length
                elif isinstance(doc, str):
                    content = doc[:1000]
                else:
                    content = str(doc)[:1000]

                doc_content += f"Document {i+1}:\n{content}\n\n"

        return Task(
            description=dedent(
                f"""
                Create a comprehensive summary of the uploaded documents using the Enhanced RAG Retrieval Tool.
                
                Task: Analyze and summarize the key information from the following documents:
                
                {doc_content}
                
                Instructions:
                1. Identify the main topics and themes across all documents
                2. Extract key facts, concepts, and important details
                3. Organize the information logically
                4. Create a summary of 250-400 words
                5. Use bullet points for key findings when appropriate
                6. Highlight any important relationships or connections between topics
                7. Make the summary accessible and informative
                
                Focus on providing value to users who will be asking questions about this content.
                """
            ),
            expected_output=dedent(
                """
                A comprehensive document summary (250-400 words) that includes:
                - Main topics and themes identified
                - Key facts and important details
                - Logical organization of information
                - Bullet points for key findings
                - Any important relationships between topics
                - Clear, accessible language suitable for Q&A context
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def research_task(self, agent, research_topic):
        """Enhanced research task for complex queries"""
        return Task(
            description=dedent(
                f"""
                Conduct comprehensive research on the topic: '{research_topic}'
                
                Use the Enhanced RAG Retrieval Tool to:
                1. Search for information from multiple angles
                2. Gather comprehensive details about the topic
                3. Look for related concepts and connections
                4. Find specific examples or case studies
                5. Identify any important nuances or considerations
                
                Research Topic: {research_topic}
                
                Provide a thorough analysis based on available information.
                """
            ),
            expected_output=dedent(
                """
                A comprehensive research report (300-500 words) including:
                - Detailed explanation of the topic
                - Key concepts and definitions
                - Relevant examples or case studies
                - Important considerations or nuances
                - Related topics or connections
                - Well-organized, informative presentation
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def comparison_task(self, agent, items_to_compare):
        """Task for comparing multiple items or concepts"""
        return Task(
            description=dedent(
                f"""
                Compare and contrast the following items using information from the knowledge base: {items_to_compare}
                
                Instructions:
                1. Use the Enhanced RAG Retrieval Tool to gather information about each item
                2. Identify similarities and differences
                3. Create a structured comparison
                4. Highlight key distinguishing features
                5. Provide examples where relevant
                6. Organize the comparison clearly
                
                Focus on providing actionable insights from the comparison.
                """
            ),
            expected_output=dedent(
                """
                A structured comparison (250-400 words) that includes:
                - Clear identification of items being compared
                - Key similarities between items
                - Important differences and distinctions
                - Relevant examples or use cases
                - Well-organized presentation (table format if appropriate)
                - Actionable insights from the comparison
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def explanation_task(self, agent, concept_to_explain):
        """Task for explaining complex concepts"""
        return Task(
            description=dedent(
                f"""
                Provide a clear, comprehensive explanation of: '{concept_to_explain}'
                
                Use the Enhanced RAG Retrieval Tool to:
                1. Find detailed information about the concept
                2. Look for definitions and key characteristics
                3. Search for examples and applications
                4. Find related concepts or background information
                5. Identify any common misconceptions
                
                Make the explanation accessible while being thorough.
                """
            ),
            expected_output=dedent(
                """
                A clear, comprehensive explanation (250-400 words) that includes:
                - Definition and key characteristics
                - Background context where relevant
                - Practical examples or applications
                - Related concepts or connections
                - Any important nuances or considerations
                - Accessible language suitable for the intended audience
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )
