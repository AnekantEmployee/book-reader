# rag_system/tools.py
from crewai_tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import MistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM and Embeddings
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llm = MistralAI(mistral_api_key=mistral_api_key)
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)

class RagRetrievalTool(BaseTool):
    name: str = "RAG Retrieval Tool"
    description: str = "A tool to retrieve relevant documents from the vector store based on a query."

    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store

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
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Invoke the chain with the query
        response = retrieval_chain.invoke({"input": query})
        
        return response['answer']

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "A tool to perform a web search for general queries."

    # Using the pre-built SerperDevTool from crewai-tools
    # SerperDevTool needs SERPER_API_KEY environment variable
    def _run(self, query: str):
        return self.serper_dev_tool._run(query)

    def __init__(self):
        super().__init__()
        from crewai_tools import SerperDevTool
        self.serper_dev_tool = SerperDevTool()