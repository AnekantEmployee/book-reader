# rag_system/crew_setup.py
import os
from crewai import Crew, Process
from rag_system.agents import RagAgents
from rag_system.tasks import RagTasks
from rag_system.tools import RagRetrievalTool
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

# Disable CrewAI telemetry to prevent trace prompts
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

load_dotenv()


def create_rag_crew(user_query, vector_store):
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    # Create the tool with proper initialization
    qna_tool = RagRetrievalTool(vector_store=vector_store)

    agents = RagAgents([qna_tool])
    tasks = RagTasks(qna_tool)

    rag_assistant = agents.qna_agent()
    answer_task = tasks.answer_question_task(rag_assistant, user_query)

    rag_crew = Crew(
        agents=[rag_assistant],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
        share_crew=False,  # Disable sharing
        full_output=False,  # Disable full output
        step_callback=None,  # Disable callbacks
    )

    return rag_crew


def initialize_vector_store(docs):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if mistral_api_key is None:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")

    embeddings = MistralAIEmbeddings(api_key=mistral_api_key)

    # Enhanced vector store with better similarity search
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"},  # Better semantic similarity
    )

    return vector_store
