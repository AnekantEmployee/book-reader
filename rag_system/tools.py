# rag_system/tools.py
from crewai.tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai import ChatMistralAI as MistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from typing import Any
from pydantic import BaseModel, ConfigDict

# Updated RagRetrievalTool to be Pydantic-compliant
class RagRetrievalTool(BaseTool):
    name: str = "RAG Retrieval Tool"
    description: str = "A tool to retrieve relevant documents from the vector store based on a query."
    
    # 📝 New: Use a Pydantic model_config to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 📝 New: Define the vector_store as a class field with a type hint
    vector_store: Any

    def _run(self, query: str):
        # Create a retriever
        retriever = self.vector_store.as_retriever()
        
        # Define the prompt for RAG
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        
        <context>
        {context}
        </context>
        
        Question: {input}""")
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(MistralAI(), prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Invoke the chain with the query
        response = retrieval_chain.invoke({"input": query})
        
        return response['answer']