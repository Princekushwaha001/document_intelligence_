"""
prompt.py
---------
RAG system prompts and prompt templates.

This module contains all prompt templates used in the RAG pipeline,
keeping prompt engineering separate from the core retrieval logic.
"""


# ---------------------------------------------------------------------------
# Grounding system prompt
# The model receives this before every query. It enforces that answers
# come only from the retrieved context — not from the model's training data.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful Q&A assistant. You must follow these rules strictly:

1. Answer the question ONLY using the information in the context sections below.
2. Do NOT use your general knowledge or anything outside of the provided context.
3. If the context does not contain enough information to answer the question, you MUST respond with this exact sentence:
   "I'm sorry, but the provided documents do not contain information to answer this query."
4. Be clear and concise in your answers.
"""


def build_rag_prompt(context_text: str, user_query: str) -> str:
    """
    Build the complete RAG prompt by combining system instructions,
    retrieved context, and the user's question.

    Args:
        context_text (str): Formatted context sections from retrieved documents
        user_query (str):   The user's natural-language question

    Returns:
        str: The complete prompt ready to send to the LLM

    Example:
        >>> context = "[Context 1]\nSome relevant text..."
        >>> query = "What is annual leave?"
        >>> prompt = build_rag_prompt(context, query)
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Context from documents ---\n"
        f"{context_text}\n\n"
        f"--- Question ---\n"
        f"{user_query}"
    )