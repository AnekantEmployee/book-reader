# rag_system/crew_setup.py
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from rag_system.tools import RagRetrievalTool
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def create_rag_crew(user_query, vector_store):
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    # 📝 Fix: Pass vector_store as a keyword argument
    qna_tool = RagRetrievalTool(vector_store=vector_store)
    agents = RagAgents([qna_tool])
    tasks = RagTasks(qna_tool)

    rag_assistant = agents.qna_agent()
    answer_task = tasks.answer_question_task(rag_assistant, user_query)

    rag_crew = Crew(
        agents=[rag_assistant],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=True,
    )
    
    return rag_crew

def initialize_vector_store(docs):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    embeddings = MistralAIEmbeddings(api_key=mistral_api_key)
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
    print("Vector store initialized and persisted.")
    return vector_store