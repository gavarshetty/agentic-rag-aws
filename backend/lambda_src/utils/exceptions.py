"""Custom exception classes for the RAG system."""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize RAG error.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class BedrockServiceError(RAGError):
    """Exception raised when Bedrock service operations fail."""
    pass


class KnowledgeBaseError(RAGError):
    """Exception raised when knowledge base operations fail."""
    pass


class ValidationError(RAGError):
    """Exception raised when input validation fails."""
    pass


class ConversationServiceError(RAGError):
    """Exception raised when conversation service operations fail."""
    pass

