# rag_system/tools.py

import re
import os
import time
from crewai.tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Type, List
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
import warnings
from rag_system.config import llm, embeddings # Import LLM and embeddings from config

# Disable telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"
warnings.filterwarnings("ignore")
load_dotenv()


class RagRetrievalToolInput(BaseModel):
    query: str = Field(description="The search query to retrieve relevant documents")


class EnhancedRagRetrievalTool(BaseTool):
    name: str = "Enhanced_RAG_Retrieval_Tool"
    description: str = (
        "An enhanced tool to retrieve relevant documents from the vector store with "
        "improved semantic search and robust error handling."
    )

    args_schema: Type[BaseModel] = RagRetrievalToolInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Any

    def preprocess_query(self, query: str) -> List[str]:
        """Enhanced query preprocessing with multiple search strategies"""
        queries = [query]
        try:
            cleaned_query = re.sub(r"[^\w\s-]", " ", query.lower())
            cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
            if cleaned_query != query.lower():
                queries.append(cleaned_query)
            words = cleaned_query.split()
            if len(words) > 3:
                stop_words = {
                    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                    "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
                    "has", "had", "do", "does", "did", "will", "would", "could", "should",
                }
                key_terms = [
                    word for word in words if word not in stop_words and len(word) > 2
                ]
                if key_terms:
                    queries.append(" ".join(key_terms[:5]))
            patterns = {
                r"\bwhat is\b": "",
                r"\bexplain\b": "",
                r"\btell me about\b": "",
                r"\bhow to\b": "process",
                r"\bsteps for\b": "procedure",
            }
            for pattern, replacement in patterns.items():
                modified_query = re.sub(
                    pattern, replacement, query, flags=re.IGNORECASE
                ).strip()
                if modified_query and modified_query != query:
                    queries.append(modified_query)
        except Exception as e:
            print(f"Query preprocessing error: {e}")
        return list(set(queries))[:3]

    def _run(self, query: str) -> str:
        """Enhanced retrieval with proper LLM and embedding specification"""
        try:
            if not self.vector_store:
                return "Error: Vector store not initialized."

            query_variants = self.preprocess_query(query)
            all_results = []
            seen_content = set()

            for variant in query_variants:
                try:
                    for k in [4, 6]:
                        try:
                            results = self.vector_store.similarity_search(variant, k=k)
                            for doc in results:
                                content = doc.page_content[:500]
                                if content not in seen_content:
                                    all_results.append(doc)
                                    seen_content.add(content)
                        except Exception:
                            continue
                    try:
                        scored_results = self.vector_store.similarity_search_with_score(
                            variant, k=4
                        )
                        for doc, score in scored_results:
                            if score < 1.5:
                                content = doc.page_content[:500]
                                if content not in seen_content:
                                    all_results.append(doc)
                                    seen_content.add(content)
                    except Exception:
                        continue
                except Exception:
                    continue

            if not all_results:
                try:
                    all_results = self.vector_store.similarity_search(query, k=4)
                except Exception as e:
                    return f"Error: Could not retrieve documents - {str(e)}"

            if not all_results:
                return "No relevant information found in the knowledge base."

            all_results = all_results[:6]
            try:
                prompt_template = ChatPromptTemplate.from_template(
                    """
                You are an intelligent assistant providing accurate answers based on the provided context.
                
                Context Information:
                {context}
                
                User Question: {input}
                
                Instructions:
                1. Provide a clear, helpful answer based on the context
                2. Keep the answer between 150-300 words
                3. Use specific details from the context when available
                4. If the context doesn't fully answer the question, say so clearly
                5. Structure your response with main points
                6. Be direct and informative
                
                Answer:
                """
                )
                retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": min(4, len(all_results))},
                )
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start_time = time.time()
                response = retrieval_chain.invoke({"input": query})
                execution_time = time.time() - start_time
                if execution_time > 60:
                    print(f"Long execution time: {execution_time:.2f}s")
                return response.get("answer", "Could not generate response.")
            except Exception as e:
                error_msg = str(e)
                if "Server disconnected" in error_msg:
                    return "The server connection was interrupted. Please try again."
                elif "timeout" in error_msg.lower():
                    return "The request timed out. Please try a shorter question."
                else:
                    return self._fallback_response(all_results, query)
        except Exception as e:
            return f"Error in retrieval tool: {str(e)}"

    def _fallback_response(self, documents: List, query: str) -> str:
        """Enhanced fallback response that actually uses the retrieved context"""
        try:
            if not documents:
                return "No relevant information found in the documents."
            context_parts = []
            for i, doc in enumerate(documents[:4]):
                content = doc.page_content.strip()
                if content:
                    content = content.replace("\n\n", "\n").replace("  ", " ")
                    context_parts.append(f"Source {i+1}: {content[:800]}")
            if not context_parts:
                return "Documents were found but appear to contain no readable content."
            combined_context = "\n\n".join(context_parts)
            response = f"""Based on the available information from your documents:

{combined_context}

**Query:** "{query}"

The documents contain relevant information as shown above. While this may not directly answer your specific question about "{query}", the available content provides related information that may be helpful."""
            return response
        except Exception as e:
            return f"Found documents but encountered processing error: {str(e)}"


# Maintain backward compatibility
class RagRetrievalTool(EnhancedRagRetrievalTool):
    """Backward compatible version"""

    name: str = "RAG_Retrieval_Tool"
    description: str = (
        "A tool to retrieve relevant documents from the vector store based on a query."
    )