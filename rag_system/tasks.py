from crewai import Task
from textwrap import dedent


class RagTasks:

    def __init__(self, qna_tool):
        self.qna_tool = qna_tool

    def extract_subtopics_task(self, agent, documents):
        """
        Task to extract a table of contents or key subtopics from documents.
        """
        # Combine the content from the first few documents to analyze their structure
        doc_content = ""
        for i, doc in enumerate(documents[:3]): # Limit to a few docs for context window
            if hasattr(doc, "page_content"):
                doc_content += f"Document {i+1} Content:\n{doc.page_content}\n\n"
            elif isinstance(doc, str):
                doc_content += f"Document {i+1} Content:\n{doc}\n\n"

        return Task(
            description=dedent(
                f"""
                Analyze the provided document content and identify key headings, sections, or subtopics
                that could serve as a table of contents or suggested questions for a user.
                
                Document Content:
                {doc_content}
                
                Instructions:
                1. Read through the content to understand its structure and main themes.
                2. Extract 5-10 distinct subtopics or section titles.
                3. Present these subtopics as a bulleted list.
                4. Do not provide any additional text or explanation, just the list.
                """
            ),
            expected_output=dedent(
                """
                A bulleted list of 5-10 suggested subtopics or questions, formatted in Markdown.
                Example:
                - What is the main purpose of the document?
                - Explain the process of data collection.
                - What are the key findings?
                """
            ),
            agent=agent,
            tools=[self.qna_tool] if self.qna_tool else [],
        )

    def answer_question_task(self, agent, question):
        """Enhanced question answering task with better context handling"""
        return Task(
            description=dedent(
                f"""
                Use the Enhanced RAG Retrieval Tool to find relevant information and answer this question:
                
                Question: '{question}'
                
                CRITICAL INSTRUCTIONS:
                1. ALWAYS use the tool to retrieve context first
                2. If you get ANY relevant information, use it to provide a helpful answer
                3. If the retrieved context is related but doesn't directly answer the question, say so clearly
                4. Only say "No relevant information found" if the tool returns truly empty or unrelated results
                5. When information is partial, clearly explain what you know and what's missing
                6. Be helpful and informative with whatever context you receive
                
                Example: If asked "difference between A and B" but context shows "A vs C", explain what you know about A and mention that B information wasn't found.
                """
            ),
            expected_output=dedent(
                """
                A comprehensive answer based on the retrieved context. If context is available, provide a detailed response.
                If context is partial, provide what information is available and note limitations.
                """
            ),
            agent=agent,
            tools=[self.qna_tool],
        )

    def format_answer_task(self, agent):
        return Task(
            description=dedent(
                """
                Take the raw answer from the previous task and re-format it into a structured, easy-to-read response.
                
                Instructions:
                1. Divide the answer into clear sections using Markdown headings (##).
                2. Use bullet points (-) for lists of items or key points.
                3. The final response should be professional, well-organized, and easy to read.
                4. Do not add any new information.
                5. If the previous answer mentioned limitations or partial information, preserve that context.
                """
            ),
            expected_output=dedent(
                """
                A well-structured, formatted response using Markdown with headings and bullet points.
                """
            ),
            agent=agent,
        )

    def summarize_documents_task(self, agent, documents):
        """Enhanced document summarization task"""
        # Prepare document content for summarization
        doc_content = ""
        if documents:
            for i, doc in enumerate(documents[:5]):
                if hasattr(doc, "page_content"):
                    content = doc.page_content[:1000]
                elif isinstance(doc, str):
                    content = doc[:1000]
                else:
                    content = str(doc)[:1000]

                doc_content += f"Document {i+1}:\n{content}\n\n"

        return Task(
            description=dedent(
                f"""
                Create a comprehensive summary of the uploaded documents.
                
                Task: Analyze and summarize the key information from the following documents:
                
                {doc_content}
                
                Instructions:
                1. Identify the main topics and themes.
                2. Extract key facts, concepts, and important details.
                3. Organize the information logically.
                4. Create a summary of 250-400 words.
                5. Use bullet points for key findings when appropriate.
                6. Make the summary accessible and informative.
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