"""Lambda handler for knowledge base document ingestion."""

import json
from typing import Any, Dict

from aws_lambda_powertools.utilities.typing import LambdaContext

from ..services.bedrock_service import BedrockService
from ..utils.exceptions import KnowledgeBaseError
from ..utils.logger import get_logger


def handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:
    """
    Lambda handler for S3 upload events that trigger knowledge base ingestion.
    
    This handler is triggered when documents are uploaded to the S3 bucket.
    It starts a Bedrock Knowledge Base ingestion job to process the new documents.
    
    Args:
        event: S3 event containing bucket and object information
        context: Lambda context
        
    Returns:
        Dict with statusCode and body indicating success or failure
        
    Raises:
        KnowledgeBaseError: If ingestion job fails (allows S3 to retry)
        Exception: For any other unexpected errors (allows S3 to retry)
    """
    request_logger = get_logger(context)
    request_logger.info("Knowledge base ingestion handler invoked")
    
    try:
        # Parse S3 event
        s3_records = event.get("Records", [])
        if not s3_records:
            request_logger.warning("No S3 records found in event")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No records to process"}),
            }
        
        bedrock_service = BedrockService()
        
        # Process each S3 record
        results = []
        for record in s3_records:
            bucket_name = record["s3"]["bucket"]["name"]
            object_key = record["s3"]["object"]["key"]
            
            request_logger.info(
                f"Processing S3 upload: s3://{bucket_name}/{object_key}"
            )
            
            # Start ingestion job
            # If this fails, exception will bubble up and fail the Lambda (allowing S3 retry)
            response = bedrock_service.start_ingestion_job()
            ingestion_job = response.get("ingestionJob", {})
            job_id = ingestion_job.get("ingestionJobId")
            status = ingestion_job.get("status")
            
            results.append({
                "bucket": bucket_name,
                "key": object_key,
                "ingestion_job_id": job_id,
                "status": status,
                "success": True,
            })
            
            request_logger.info(
                f"Started ingestion job {job_id} for {object_key} "
                f"(status: {status})"
            )
        
        # Return success response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Ingestion jobs processed",
                "results": results,
            }),
        }
        
    except KnowledgeBaseError as e:
        # Log and re-raise to fail Lambda (allows S3 to retry)
        request_logger.error(
            f"Failed to start ingestion job: {e.message}",
            extra={"details": e.details},
            exc_info=True,
        )
        raise
        
    except Exception as e:
        # Log and re-raise to fail Lambda (allows S3 to retry)
        request_logger.error(f"Unexpected error in knowledge base handler: {str(e)}", exc_info=True)
        raise



