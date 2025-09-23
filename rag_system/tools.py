# rag_system/tools.py
import re
import os
from crewai.tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Type, List
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

# Disable telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()


class RagRetrievalToolInput(BaseModel):
    query: str = Field(description="The search query to retrieve relevant documents")


class RagRetrievalTool(BaseTool):
    name: str = "RAG_Retrieval_Tool"
    description: str = (
        "A tool to retrieve relevant documents from the vector store based on a query."
    )
    args_schema: Type[BaseModel] = RagRetrievalToolInput

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Any

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query to handle variations and improve semantic matching"""
        queries = [query]  # Original query

        # Handle hyphen/space variations
        if "-" in query:
            queries.append(query.replace("-", " "))
        if " " in query:
            queries.append(query.replace(" ", "-"))
            queries.append(query.replace(" ", ""))

        # Handle case variations
        queries.append(query.lower())
        queries.append(query.upper())
        queries.append(query.title())

        # Split compound words and acronyms
        words = re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", query)
        if len(words) > 1:
            queries.extend([" ".join(words), "".join(words)])

        # Remove duplicates while preserving order
        seen = set()
        return [q for q in queries if not (q in seen or seen.add(q))]

    def _run(self, query: str) -> str:
        """Retrieve information from vector store with enhanced semantic search"""
        try:
            if not self.vector_store:
                return "Vector store is not initialized. Please upload documents first."

            # Generate query variations for better matching
            query_variations = self.preprocess_query(query)

            # Try multiple retrieval strategies
            all_docs = []
            for q in query_variations[:3]:  # Limit to top 3 variations
                try:
                    retriever = self.vector_store.as_retriever(
                        search_type="mmr",  # Maximum Marginal Relevance
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
                    )
                    docs = retriever.get_relevant_documents(q)
                    all_docs.extend(docs)
                    if docs:  # If we found documents, we can stop
                        break
                except:
                    continue

            if not all_docs:
                # Fallback: Try similarity search with lower threshold
                try:
                    docs = self.vector_store.similarity_search(query_variations[0], k=5)
                    all_docs = docs
                except:
                    pass

            if not all_docs:
                return f"I couldn't find specific information about '{query}'. Please try rephrasing your question or ensure the relevant documents are uploaded."

            # Enhanced prompt for better understanding
            prompt = ChatPromptTemplate.from_template(
                """
            You are a helpful assistant that answers questions based on the provided context.
            
            Context: {context}
            
            Question: {input}
            
            Instructions:
            - Provide a concise answer (150-200 words maximum)
            - Be direct and informative
            - If the context contains relevant information, use it to answer
            - If the query uses different spelling/formatting (like "fav-up" vs "fav up"), understand they refer to the same concept
            - Focus on the core meaning and intent of the question
            - Don't mention that you're using context or tools
            
            Answer:
            """
            )

            # Fix: Use proper Mistral model initialization with API key
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            llm = ChatMistralAI(
                model="mistral-small-latest",  # Correct model name
                mistral_api_key=mistral_api_key,
                temperature=0.1,
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(
                self.vector_store.as_retriever(search_kwargs={"k": 5}), document_chain
            )

            response = retrieval_chain.invoke(
                {"input": f"What is {query}? Explain about {query}."}
            )

            return response["answer"]

        except Exception as e:
            return f"I encountered an issue while searching for information about '{query}'. Error: {str(e)}"
