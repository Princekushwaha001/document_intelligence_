from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS

from app.indexer import build_vector_store
from app.rag import get_rag_answer
from app.schemas import QueryRequest, QueryResponse


# This dictionary holds the vector store in memory while the server is running.
# It is populated at startup and shared across all requests.
app_state: dict = {}

DATA_FOLDER = "data"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once when the server starts.
    Loads and indexes all documents from the data/ folder into memory.
    """
    print("Server starting up... indexing documents from data/ folder.")

    try:
        vector_store: FAISS = await build_vector_store(DATA_FOLDER)
        app_state["vector_store"] = vector_store
        print("Startup complete. Server is ready to accept requests.")
    except Exception as e:
        print(f"ERROR during startup: {e}")
        raise

    yield  # Server runs here

    print("Server shutting down.")
    app_state.clear()


# Create the FastAPI app with the lifespan handler
app = FastAPI(
    title="Document Intelligence API",
    description="A RAG-based Q&A bot that only answers from local documents.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    POST /v1/query

    Accepts a user question, searches the indexed documents,
    and returns a grounded answer.

    Request body:
        { "user_query": "What is the policy on annual leave?" }
    """
    vector_store: FAISS = app_state.get("vector_store")

    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready. Please try again shortly.")

    try:
        response: QueryResponse = await get_rag_answer(
            user_query=request.user_query,
            vector_store=vector_store,
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your query: {str(e)}")


@app.get("/health")
async def health_check() -> dict:
    """Simple health check to confirm the server and index are running."""
    if "vector_store" not in app_state:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    return {"status": "ok"}
