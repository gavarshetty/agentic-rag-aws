"""Configuration and environment variable management."""

import os
from typing import Optional
from aws_lambda_powertools import Logger

logger = Logger(service="agentic-rag-config")


class Config:
    """Application configuration from environment variables."""
    
    # Knowledge Base Configuration
    KNOWLEDGE_BASE_ID: str
    S3_BUCKET_NAME: str
    
    # Model Configuration
    DEFAULT_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    FALLBACK_MODEL_ID: str = "meta.llama3-1-8b-instruct-v1:0"
    
    # AWS Region
    AWS_REGION: str
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Required environment variables
        self.KNOWLEDGE_BASE_ID = self._get_env_var("KNOWLEDGE_BASE_ID", required=True)
        self.S3_BUCKET_NAME = self._get_env_var("S3_BUCKET_NAME", required=True)
        
        
        # AWS Region (required for Bedrock operations)
        self.AWS_REGION = self._get_env_var("AWS_REGION", default=self._get_aws_region(), required=True)
        
    def _get_env_var(self, name: str, default: Optional[str] = None, required: bool = False) -> str:
        """
        Get environment variable with optional default.
        
        Args:
            name: Environment variable name
            default: Optional default value
            required: Whether the variable is required
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is missing
        """
        value = os.getenv(name, default)
        
        if required and not value:
            raise ValueError(f"Required environment variable {name} is not set")
        
        if value:
            logger.debug(f"Config: {name} = {value}")
        
        return value or ""
    
    def _get_aws_region(self) -> str:
        """
        Get AWS region from environment or boto3 session.
        
        Returns:
            AWS region string
        """
        # Try environment variable first
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        
        if region:
            return region
        
        # Try to get from boto3 session
        try:
            import boto3
            session = boto3.Session()
            return session.region_name or "us-east-1"
        except Exception:
            # Fallback to default
            return "us-east-1"
    
    def get_model_arn(self, model_id: Optional[str] = None) -> str:
        """
        Get full ARN for a model ID.
        
        Args:
            model_id: Optional model ID. Uses default if not provided.
            
        Returns:
            Full model ARN
        """
        model = model_id or self.DEFAULT_MODEL_ID
        return f"arn:aws:bedrock:{self.AWS_REGION}::foundation-model/{model}"


# Global config instance
config = Config()

