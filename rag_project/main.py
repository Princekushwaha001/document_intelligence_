
"""
main.py
-------
FastAPI application entry point for the Document Intelligence API.

Responsibility:
    - Create the FastAPI app instance.
    - Define the lifespan handler that builds the FAISS index at startup
      and clears it gracefully on shutdown.
    - Register all HTTP endpoints:
        POST /v1/query  — the main RAG query endpoint
        GET  /health    — simple readiness check

How the server starts:
    Run from the project root with:
        uvicorn main:app --reload

Flow on startup:
    lifespan() is called
        → build_vector_store("data") loads, chunks, embeds all documents
        → FAISS index stored in app_state["vector_store"] (RAM only)
        → server begins accepting requests

Flow on each POST /v1/query request:
    query() endpoint called
        → validates request with QueryRequest (Pydantic)
        → calls get_rag_answer() in rag.py
        → returns QueryResponse (Pydantic)

Flow on shutdown:
    lifespan() resumes after yield
        → app_state.clear() removes the index from memory
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS

from app.indexer import build_vector_store
from app.rag import get_rag_answer
from app.schemas import QueryRequest, QueryResponse


# ---------------------------------------------------------------------------
# In-memory application state
# The FAISS vector store is built once at startup and shared across all
# requests via this dictionary. Nothing here is persisted to disk.
# ---------------------------------------------------------------------------

app_state: dict = {}

DATA_FOLDER = "data"

# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown logic
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Async context manager that manages the application lifespan.

    Startup (code before yield):
        - Calls build_vector_store() to load, chunk, embed and index all
          documents found in the DATA_FOLDER directory.
        - Stores the resulting FAISS index in app_state["vector_store"].
        - If indexing fails, the server raises immediately and does not start.

    Shutdown (code after yield):
        - Clears app_state to release the FAISS index from memory.

    Args:
        app (FastAPI): The FastAPI application instance (provided by FastAPI).

    Yields:
        None: Control returns to FastAPI while the server is running.
    """

    print("Server starting up... indexing documents from data/ folder.")

    try:
        vector_store: FAISS = await build_vector_store(DATA_FOLDER)
        app_state["vector_store"] = vector_store
        print("Startup complete. Server is ready to accept requests.")
    except Exception as e:
        print(f"ERROR during startup: {e}")
        raise

    yield  # ← server is live and handling requests here

    print("Server shutting down.")
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Intelligence API",
    description="A RAG-based Q&A bot that only answers from local documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Accept a natural-language question and return a grounded answer.

    This endpoint runs the full RAG pipeline:
        1. Validate the request body with Pydantic (QueryRequest).
        2. Retrieve the top-4 relevant chunks from the FAISS index.
        3. Pass those chunks + the question to the Groq LLM.
        4. Return the answer and the source documents (QueryResponse).

    The LLM is strictly prompted to answer ONLY from the retrieved
    document context. If the context is insufficient, it returns:
        "I'm sorry, but the provided documents do not contain
         information to answer this query."

    Args:
        request (QueryRequest): Pydantic model containing 'user_query'.

    Returns:
        QueryResponse: Contains 'answer' (str) and 'sources' (list).

    Raises:
        HTTPException 503: If the vector store has not been initialised yet.
        HTTPException 500: If the RAG pipeline raises an unexpected error.

    Example request body:
        { "user_query": "What is the policy on annual leave?" }

    Example response:
        {
            "answer": "Employees are entitled to 20 days of annual leave.",
            "sources": [
                {
                    "source": "data/company_handbook.txt",
                    "excerpt": "Full-time employees are entitled to 20 days..."
                }
            ]
        }
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
    """
    Readiness check endpoint.

    Returns HTTP 200 with {"status": "ok"} when the server is running
    and the FAISS index has been successfully built.

    Returns HTTP 503 if the index is not ready (e.g. still starting up).

    Returns:
        dict: {"status": "ok"} when the server is healthy.

    Raises:
        HTTPException 503: If the vector store is not yet initialised.
    """
    
    if "vector_store" not in app_state:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    return {"status": "ok"}
