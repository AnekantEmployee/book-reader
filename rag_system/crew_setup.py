# rag_system/crew_setup.py
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from rag_system.tools import RagRetrievalTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Global vector store
vector_store = None

def create_rag_crew(user_query):
    # This assumes that the vector store is loaded in the main app
    global vector_store
    if vector_store is None:
        raise ValueError("Vector store has not been initialized. Please upload documents first.")

    # Initialize tools with the loaded vector store
    qna_tool = RagRetrievalTool(vector_store)

    # Initialize agents and tasks
    agents = RagAgents([qna_tool])
    tasks = RagTasks(qna_tool)

    rag_assistant = agents.qna_agent()
    answer_task = tasks.answer_question_task(rag_assistant, user_query)

    # Create the crew
    rag_crew = Crew(
        agents=[rag_assistant],
        tasks=[answer_task],
        process=Process.sequential,
        manager_llm=os.getenv("MISTRAL_API_KEY"),
        verbose=2,
    )
    
    return rag_crew

def initialize_vector_store(docs):
    global vector_store
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
    # Using ChromaDB as a file-based persistent vector store
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    print("Vector store initialized and persisted.")