"""Request models for API input validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="The user's query/question", min_length=1)
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID for maintaining context"
    )
    history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional conversation history as list of message dicts with 'role' and 'content'",
    )
    model_id: Optional[str] = Field(
        None,
        description="Optional model ID to use. Defaults to Claude 3 Haiku if not specified",
    )
    max_results: Optional[int] = Field(
        5, description="Maximum number of retrieval results to use", ge=1, le=10
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "query": "What is the main topic discussed in the documents?",
                "conversation_id": "conv-123",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"},
                ],
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "max_results": 5,
            }
        }

        extra = "forbid"

