
"""
schemas.py
----------
Pydantic models that define the shape of API request and response data.

Responsibility:
    - Validate incoming request payloads automatically (FastAPI uses these).
    - Define the exact JSON structure the client receives in responses.
    - Act as living documentation for the API contract.

Models:
    QueryRequest     — body accepted by POST /v1/query
    SourceDocument   — one retrieved document chunk inside the response
    QueryResponse    — full response returned by POST /v1/query
"""

from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for POST /v1/query.

    Attributes:
        user_query (str): The natural-language question the user wants answered.
                          Must be at least 1 character long.

    Example JSON:
        {
            "user_query": "What is the policy on annual leave?"
        }
    """

    user_query: str = Field(
        ..., 
        min_length=1, 
        description="The question to ask"
    )


class SourceDocument(BaseModel):
    """
    Represents a single document chunk that was retrieved from the vector
    store and used to ground the LLM's answer.

    Attributes:
        source (str):  The filename of the source document (e.g. 'company_handbook.txt').
        excerpt (str): The first 200 characters of the retrieved chunk,
                       shown so the user can verify where the answer came from.
    """

    source: str = Field(..., description="The filename of the source document")
    excerpt: str = Field(..., description="A short excerpt from the relevant chunk")


class QueryResponse(BaseModel):
    """
    Response body returned by POST /v1/query.

    Attributes:
        answer (str):                  The grounded answer produced by the LLM.
                                       If the documents do not contain the answer,
                                       this will be the fixed fallback sentence.
        sources (List[SourceDocument]): The document chunks the LLM used to
                                        generate the answer.

    Example JSON:
        {
            "answer": "Employees are entitled to 20 days of annual leave.",
            "sources": [
                {
                    "source": "company_handbook.txt",
                    "excerpt": "Full-time employees are entitled to 20 days..."
                }
            ]
        }
    """

    answer: str = Field(..., description="The answer from the LLM, grounded in the documents")
    sources: List[SourceDocument] = Field(..., description="The document chunks used to generate the answer")
