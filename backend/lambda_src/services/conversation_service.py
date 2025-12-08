"""Conversation service for managing chat history."""

import uuid
import boto3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils.logger import get_logger
from ..utils.exceptions import ConversationServiceError
from ..utils.config import config

logger = get_logger()


class ConversationService:
    """
    Service for managing conversation context and history.
    
    Uses DynamoDB for persistent storage across Lambda invocations.
    """

    def __init__(self):
        """Initialize conversation service with DynamoDB client."""
        self.dynamodb = boto3.resource('dynamodb', region_name=config.AWS_REGION)
        self.table = self.dynamodb.Table(config.CONVERSATIONS_TABLE_NAME)
        logger.debug(f"Initialized ConversationService with table: {config.CONVERSATIONS_TABLE_NAME}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying DynamoDB put_item for message (attempt {retry_state.attempt_number})"
        ),
    )
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to a conversation.
        
        Creates a new message record with epoch timestamp as message_id (sort key).
        Retries on transient DynamoDB errors (throttling, service unavailable).
        
        Args:
            conversation_id: The conversation ID
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata (e.g., sources, model_used)
            
        Raises:
            ConversationServiceError: If DynamoDB operation fails
        """
        now = datetime.now(timezone.utc)
        # Generate message_id as epoch timestamp in microseconds
        message_id = int(now.timestamp() * 1000000)
        created_at = now.isoformat()
        ttl_timestamp = int((now + timedelta(days=1)).timestamp())
        
        message_item = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": created_at,
            "ttl": ttl_timestamp,
        }
        
        if metadata:
            message_item["metadata"] = metadata
        
        try:
            # Insert new message record
            self.table.put_item(Item=message_item)
            logger.debug(
                f"Added {role} message (message_id: {message_id}) to conversation {conversation_id}"
            )
            
            # NOTE: TTL is NOT updated on existing messages (simplified approach)
            # Each message expires 1 day after it was added, causing "memory fade"
            # where older messages gradually disappear from the conversation.
            # 
            # For production: Use TransactWriteItems to batch-update TTL on all messages
            # when a new message is added. This keeps the entire conversation alive
            # as long as it's active. See README.md for implementation details.
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Retry on transient errors
            if error_code in ['ThrottlingException', 'InternalServerException', 
                              'ServiceUnavailableException', 'ProvisionedThroughputExceededException']:
                logger.warning(f"Transient DynamoDB error: {error_code}, will retry")
                raise  # Let tenacity handle the retry
            
            # For permanent errors, raise exception
            error_msg = f"Failed to add message to conversation in DynamoDB: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationServiceError(
                error_msg,
                {"conversation_id": conversation_id, "message_id": message_id, "error_code": error_code}
            )
        
        except Exception as e:
            # Catch all other exceptions (network, serialization, programming errors, etc.)
            error_msg = f"Unexpected error adding message to conversation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationServiceError(
                error_msg,
                {"conversation_id": conversation_id, "message_id": message_id}
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying DynamoDB query for history (attempt {retry_state.attempt_number})"
        ),
    )
    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in a format suitable for model API calls.
        
        Queries all messages for the conversation, sorted chronologically by message_id.
        Retries on transient DynamoDB errors (throttling, service unavailable).
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of message dicts with 'role' and 'content' keys (empty list if no messages exist)
            
        Raises:
            ConversationServiceError: If DynamoDB operation fails
        """
        try:
            # Query all messages for this conversation, sorted by message_id (chronological order)
            response = self.table.query(
                KeyConditionExpression="conversation_id = :conv_id",
                ExpressionAttributeValues={":conv_id": conversation_id},
                ScanIndexForward=True,  # Ascending order (oldest messages first)
            )
            
            messages = response.get("Items", [])
            
            # Transform to format expected by model API calls
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Retry on transient errors
            if error_code in ['ThrottlingException', 'InternalServerException', 
                              'ServiceUnavailableException', 'ProvisionedThroughputExceededException']:
                logger.warning(f"Transient DynamoDB error: {error_code}, will retry")
                raise  # Let tenacity handle the retry
            
            # For permanent errors, raise exception
            error_msg = f"Failed to get conversation history from DynamoDB: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationServiceError(
                error_msg,
                {"conversation_id": conversation_id, "error_code": error_code}
            )
        
        except Exception as e:
            # Catch all other exceptions (network, serialization, programming errors, etc.)
            error_msg = f"Unexpected error getting conversation history: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationServiceError(
                error_msg,
                {"conversation_id": conversation_id}
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying DynamoDB query (attempt {retry_state.attempt_number})"
        ),
    )
    def ensure_conversation_exists(self, conversation_id: Optional[str] = None) -> str:
        """
        Ensure a conversation ID exists, generating a new one if needed.
        
        Conversations are created implicitly when the first message is added.
        This method just ensures we have a valid conversation_id to use.
        
        Retries on transient DynamoDB errors (throttling, service unavailable).
        
        Args:
            conversation_id: Optional conversation ID. Generates new if not provided or not found.
            
        Returns:
            Conversation ID (existing or newly generated)
        """
        if conversation_id:
            # Check if conversation exists by querying for any messages
            try:
                response = self.table.query(
                    KeyConditionExpression="conversation_id = :conv_id",
                    ExpressionAttributeValues={":conv_id": conversation_id},
                    Limit=1,  # Only need to check existence
                )
                
                if response.get("Items"):
                    # Conversation exists (has at least one message)
                    return conversation_id
                    
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                
                # Retry on transient errors
                if error_code in ['ThrottlingException', 'InternalServerException', 
                                  'ServiceUnavailableException', 'ProvisionedThroughputExceededException']:
                    logger.warning(f"Transient DynamoDB error: {error_code}, will retry")
                    raise  # Let tenacity handle the retry
                
                # For permanent errors (e.g., access denied, validation), raise exception
                error_msg = f"DynamoDB error checking conversation existence: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ConversationServiceError(
                    error_msg,
                    {"conversation_id": conversation_id, "error_code": error_code}
                )
            
            except Exception as e:
                # Catch all other exceptions (network, serialization, programming errors, etc.)
                error_msg = f"Unexpected error checking conversation existence: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ConversationServiceError(
                    error_msg,
                    {"conversation_id": conversation_id}
                )
        
        # Generate new conversation ID (conversation will be created when first message is added)
        new_conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
        logger.debug(f"Generated new conversation_id: {new_conversation_id}")
        return new_conversation_id

    def get_or_create_history(
        self, conversation_id: Optional[str] = None
    ) -> tuple[str, List[Dict[str, str]]]:
        """
        Get conversation history, generating a new conversation ID if needed.
        
        Args:
            conversation_id: Optional conversation ID
            
        Returns:
            Tuple of (conversation_id, history)
            History will be empty list if conversation has no messages yet.
        """
        conv_id = self.ensure_conversation_exists(conversation_id)
        history = self.get_history(conv_id)  # Returns empty list if no messages exist
        return conv_id, history


