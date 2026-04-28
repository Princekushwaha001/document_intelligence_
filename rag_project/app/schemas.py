from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /v1/query"""
    user_query: str = Field(..., min_length=1, description="The question to ask")


class SourceDocument(BaseModel):
    """A single document chunk that was used to answer the question"""
    source: str = Field(..., description="The filename of the source document")
    excerpt: str = Field(..., description="A short excerpt from the relevant chunk")


class QueryResponse(BaseModel):
    """Response body returned by POST /v1/query"""
    answer: str = Field(..., description="The answer from the LLM, grounded in the documents")
    sources: List[SourceDocument] = Field(..., description="The document chunks used to generate the answer")
