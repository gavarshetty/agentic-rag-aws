"""Bedrock service wrapper for AWS Bedrock operations."""

import json
import boto3
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from botocore.exceptions import ClientError

from ..utils.logger import get_logger
from ..utils.exceptions import BedrockServiceError, KnowledgeBaseError
from ..utils.config import config

logger = get_logger()


class BedrockService:
    """Service for interacting with AWS Bedrock."""

    def __init__(self, region: Optional[str] = None):
        """
        Initialize Bedrock service.
        
        Args:
            region: Optional AWS region. Uses config default if not provided.
        """
        self.region = region or config.AWS_REGION
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime", region_name=self.region
        )
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region)
        self.bedrock = boto3.client("bedrock", region_name=self.region)
        self.knowledge_base_id = config.KNOWLEDGE_BASE_ID
        self.s3_data_source_id = config.S3_DATA_SOURCE_ID

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Bedrock retrieval (attempt {retry_state.attempt_number})"
        )
    )
    def retrieve(
        self, query: str, max_results: int = 5, next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks from the knowledge base.

        Args:
            query: The search query
            max_results: Maximum number of results to return (1-10)
            next_token: Optional pagination token

        Returns:
            Dict containing Bedrock retrieval response with 'retrievalResults' key

        Raises:
            KnowledgeBaseError: If retrieval fails
        """
        try:
            logger.info(f"Retrieving from knowledge base: {self.knowledge_base_id}")

            params = {
                "knowledgeBaseId": self.knowledge_base_id,
                "retrievalQuery": {
                    "text": query,
                },
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": min(max_results, 10),
                    }
                },
            }

            if next_token:
                params["nextToken"] = next_token

            response = self.bedrock_agent_runtime.retrieve(**params)

            logger.info(f"Retrieved {len(response.get('retrievalResults', []))} results")

            return response

        except ClientError as e:
            # Check if it's a transient error worth retrying
            error_code = e.response.get('Error', {}).get('Code', '')

            # Transient errors that should be retried
            transient_errors = [
                'ThrottlingException',
                'InternalServerException',
                'ServiceUnavailableException',
                'LimitExceededException'
            ]

            if error_code in transient_errors:
                # Let tenacity handle the retry
                logger.warning(f"Transient error {error_code}, will retry: {str(e)}")
                raise e
            else:
                # Permanent error - don't retry
                logger.error(f"Permanent error {error_code}, not retrying: {str(e)}")
                raise KnowledgeBaseError(
                    f"Bedrock API error: {str(e)}",
                    {
                        "error_code": error_code,
                        "query": query,
                        "knowledge_base_id": self.knowledge_base_id
                    }
                )

        except Exception as e:
            # Non-ClientError exceptions (network, etc.) - don't retry
            error_msg = f"Failed to retrieve from knowledge base: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise KnowledgeBaseError(error_msg, {"query": query, "knowledge_base_id": self.knowledge_base_id})


    def invoke_model(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Invoke a Bedrock foundation model directly.
        
        Args:
            model_id: The model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt (Claude-style)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
            
        Raises:
            BedrockServiceError: If model invocation fails
        """
        try:
            # Determine model provider
            if "claude" in model_id.lower():
                return self._invoke_claude(
                    model_id, messages, system_prompt, temperature, max_tokens
                )
            elif "llama" in model_id.lower():
                return self._invoke_llama(model_id, messages, temperature, max_tokens)
            else:
                raise BedrockServiceError(
                    f"Unsupported model: {model_id}",
                    {"model_id": model_id},
                )
                
        except BedrockServiceError:
            raise
        except Exception as e:
            error_msg = f"Failed to invoke model {model_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise BedrockServiceError(error_msg, {"model_id": model_id})

    def _invoke_claude(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Invoke Claude model."""
        model_arn = config.get_model_arn(model_id)
        
        # "messages" contains only the most recent user query,
        # formatted as a list of dicts: [{'role': 'user', 'content': '...'}].
        #
        # system_prompt is a string that includes instructions, retrieved context,
        # and the full conversation history. It acts as the system prompt for the model.
        # Example: system_prompt = "Instructions + Retrieved Context + Full conversation"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system_prompt:
            body["system_prompt"] = system_prompt
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _invoke_llama(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Invoke Llama model."""
        body = {
            # For Llama, "messages" contains general instructions, retrieved context, 
            # and the whole conversation history, structured as a list of dicts:
            # [{'role': 'system'/'user'/'assistant', 'content': '...'}, ...].
            "messages": messages,
            "temperature": temperature,
            "max_gen_len": max_tokens,
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["generation"]

    def start_ingestion_job(
        self, data_source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a knowledge base ingestion job.
        
        Args:
            data_source_id: Optional data source ID. Uses default if not provided.
            
        Returns:
            Dict with 'ingestionJob' info including 'ingestionJobId' and 'status'
            
        Raises:
            KnowledgeBaseError: If job start fails
        """
        try:
            logger.info(f"Starting ingestion job for knowledge base: {self.knowledge_base_id}")
            
            # Get data source if not provided
            if not data_source_id:
                data_source_id = self.s3_data_source_id

            response = self.bedrock.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=data_source_id,
            )
            
            job = response.get("ingestionJob", {})
            logger.info(
                f"Ingestion job started: {job.get('ingestionJobId')} "
                f"(status: {job.get('status')})"
            )
            return response

        except Exception as e:
            error_msg = f"Failed to start ingestion job: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise KnowledgeBaseError(
                error_msg,
                {"knowledge_base_id": self.knowledge_base_id, "data_source_id": data_source_id},
            )

    def get_ingestion_job_status(self, ingestion_job_id: str, data_source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the status of an ingestion job.
        
        Args:
            ingestion_job_id: The unique identifier of the ingestion job.
            data_source_id: Optional data source ID associated with the ingestion job. If not provided, uses the default configured data source.

        Returns:
            Dict containing detailed ingestion job status and metadata as returned by AWS Bedrock.

        Raises:
            KnowledgeBaseError: If the status retrieval fails or an exception is encountered.
        """
        try:
            if not data_source_id:
                data_source_id = self.s3_data_source_id

            response = self.bedrock.get_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=ingestion_job_id,
            )
            return response.get("ingestionJob", {})
        except Exception as e:
            error_msg = f"Failed to get ingestion job status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise KnowledgeBaseError(
                error_msg, {"ingestion_job_id": ingestion_job_id}
            )

