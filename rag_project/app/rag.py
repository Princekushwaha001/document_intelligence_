from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY
from app.schemas import QueryResponse, SourceDocument


# System prompt that strictly grounds the LLM to only use the provided context.
# If the answer is not in the context, the model must use the exact fallback sentence.
SYSTEM_PROMPT = """You are a helpful Q&A assistant. You must follow these rules strictly:

1. Answer the question ONLY using the information in the context sections below.
2. Do NOT use your general knowledge or anything outside of the provided context.
3. If the context does not contain enough information to answer the question, you MUST respond with this exact sentence:
   "I'm sorry, but the provided documents do not contain information to answer this query."
4. Be clear and concise in your answers.
"""


async def get_rag_answer(user_query: str, vector_store: FAISS) -> QueryResponse:
    """
    RAG pipeline:
      1. Retrieve the top 4 most relevant chunks from the vector store
      2. Build a prompt with those chunks as context
      3. Call the LLM asynchronously
      4. Return the answer and source documents

    Args:
        user_query: The question from the user.
        vector_store: The in-memory FAISS index built at startup.

    Returns:
        A QueryResponse with the answer and source documents used.
    """
    # Step 1: Retrieve relevant chunks from FAISS
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_chunks: List[Document] = await retriever.ainvoke(user_query)

    # Step 2: Format the retrieved chunks into a context string
    context_sections = []
    for i, chunk in enumerate(relevant_chunks, start=1):
        context_sections.append(f"[Context {i}]\n{chunk.page_content.strip()}")

    context_text = "\n\n".join(context_sections)

    # Step 3: Build the full message for the LLM
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Context from documents ---\n"
        f"{context_text}\n\n"
        f"--- Question ---\n"
        f"{user_query}"
    )

    # Step 4: Call the LLM asynchronously
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,                   # Keep answers factual and consistent
        groq_api_key=GROQ_API_KEY,
    )
    response = await llm.ainvoke(full_prompt)
    answer: str = response.content

    # Step 5: Build the list of source documents for the response
    sources: List[SourceDocument] = [
        SourceDocument(
            source=chunk.metadata.get("source", "unknown"),
            excerpt=chunk.page_content.strip()[:200],
        )
        for chunk in relevant_chunks
    ]

    return QueryResponse(answer=answer, sources=sources)
