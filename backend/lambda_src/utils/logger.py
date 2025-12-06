"""Logging utilities using AWS Lambda Powertools."""

from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext


def get_logger(context: LambdaContext = None) -> Logger:
    """
    Get a configured logger instance.
    
    Args:
        context: Optional Lambda context for request ID extraction
        
    Returns:
        Configured Logger instance
    """
    logger = Logger(
        service="agentic-rag",
        level="INFO",
    )
    
    if context:
        logger.append_keys(request_id=context.request_id)
    
    return logger

