
"""
rag.py
------
Retrieval-Augmented Generation (RAG) query pipeline.

Responsibility:
    Given a user question and the in-memory FAISS vector store, this module:
        1. Retrieves the top-4 most semantically relevant document chunks.
        2. Assembles those chunks into a numbered context block.
        3. Calls the Groq LLM (llama-3.3-70b-versatile) with a strict
           grounding system prompt so the model only answers from the context.
        4. Returns the answer and the source chunks as a QueryResponse.

Grounding rule:
    The SYSTEM_PROMPT (imported from prompt.py) instructs the model that
    it MUST NOT use general world knowledge. If the retrieved context does not 
    contain enough information, the model must respond with this exact sentence:
        "I'm sorry, but the provided documents do not contain
         information to answer this query."

Public functions:
    get_rag_answer(user_query, vector_store) — full RAG pipeline
"""


from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY
from app.schemas import QueryResponse, SourceDocument
from app.prompt import build_rag_prompt



async def get_rag_answer(user_query: str, vector_store: FAISS) -> QueryResponse:
    """
    Execute the full RAG pipeline for a single user query.

    Steps:
        1. Semantic retrieval  — FAISS similarity search, top 4 chunks
        2. Context assembly    — format chunks into a numbered context block
        3. Prompt construction — combine SYSTEM_PROMPT + context + question
        4. LLM call            — async call to Groq (llama-3.3-70b-versatile)
        5. Response packaging  — wrap answer + sources in QueryResponse

    Args:
        user_query (str):       The natural-language question from the user.
        vector_store (FAISS):   The in-memory FAISS index built at startup.

    Returns:
        QueryResponse: Contains the grounded answer string and the list of
                       SourceDocument objects that were used to produce it.

    Raises:
        Exception: Any exception from the Groq API or FAISS retriever
                   is allowed to propagate up to the endpoint handler,
                   which wraps it in an HTTP 500 response.

    Example:
        >>> response = await get_rag_answer("What is annual leave?", vs)
        >>> print(response.answer)
        'Employees are entitled to 20 days of paid annual leave per year.'
    """
    # ------------------------------------------------------------------
    # Step 1 — Retrieve the top-4 most relevant chunks from FAISS
    # as_retriever() wraps the FAISS index in a standard LangChain
    # retriever interface. ainvoke() performs the search asynchronously.
    # ------------------------------------------------------------------

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_chunks: List[Document] = await retriever.ainvoke(user_query)

    # ------------------------------------------------------------------
    # Step 2 — Format chunks into a readable numbered context block
    # Each chunk is labelled [Context N] so the model can reference them.
    # ------------------------------------------------------------------

    context_sections = []
    for i, chunk in enumerate(relevant_chunks, start=1):
        context_sections.append(f"[Context {i}]\n{chunk.page_content.strip()}")

    context_text = "\n\n".join(context_sections)

    # ------------------------------------------------------------------
    # Step 3 — Build the full prompt using the prompt builder function
    # This now calls build_rag_prompt() from prompt.py which handles
    # combining the system prompt, context, and user question.
    # ------------------------------------------------------------------
 
    full_prompt = build_rag_prompt(context_text, user_query)

    # ------------------------------------------------------------------
    # Step 4 — Call the Groq LLM asynchronously
    # temperature=0 makes responses deterministic and factual.
    # ainvoke() sends the request without blocking the event loop.
    # ------------------------------------------------------------------

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,                   # Keep answers factual and consistent
        groq_api_key=GROQ_API_KEY,
    )
    response = await llm.ainvoke(full_prompt)
    answer: str = response.content

    # ------------------------------------------------------------------
    # Step 5 — Build the source list for the response
    # Expose the first 200 characters of each chunk as an excerpt so
    # the user can see exactly where the answer came from.
    # ------------------------------------------------------------------

    sources: List[SourceDocument] = [
        SourceDocument(
            source=chunk.metadata.get("source", "unknown"),
            excerpt=chunk.page_content.strip()[:200],
        )
        for chunk in relevant_chunks
    ]

    return QueryResponse(answer=answer, sources=sources)
