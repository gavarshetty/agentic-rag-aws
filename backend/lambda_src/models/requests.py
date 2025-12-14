"""Request models for API input validation."""

from typing import Optional
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="The user's query/question", min_length=1)
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID for maintaining context"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "query": "What is the main topic discussed in the documents?",
                "conversation_id": "conv-123",
            }
        }

        extra = "forbid"

