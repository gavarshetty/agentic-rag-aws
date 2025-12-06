"""Response models for API output."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Standardized response format for RAG queries."""

    response: str = Field(..., description="The generated response from the model")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of retrieved document chunks with metadata (location, score, etc.)",
    )
    conversation_id: str = Field(..., description="The conversation ID for this query")
    model_used: str = Field(..., description="The model ID that was used for generation")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "response": "Based on the documents, the main topic is...",
                "sources": [
                    {
                        "location": {
                            "s3Location": {
                                "uri": "s3://bucket/documents/doc1.pdf"
                            },
                            "type": "S3",
                        },
                        "score": 0.95,
                    }
                ],
                "conversation_id": "conv-123",
                "model_used": "anthropic.claude-3-haiku-20240307-v1:0",
            }
        }

