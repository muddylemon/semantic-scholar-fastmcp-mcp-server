"""
Error handling utilities for the Semantic Scholar API Server.
"""

from typing import Dict, Optional
from ..config import ErrorType

def create_error_response(
    error_type: ErrorType,
    message: str,
    details: Optional[Dict] = None
) -> Dict:
    """
    Create a standardized error response.

    Args:
        error_type: The type of error that occurred.
        message: A human-readable message describing the error.
        details: Optional additional details about the error.

    Returns:
        A dictionary with the error information.
    """
    return {
        "error": {
            "type": error_type.value,
            "message": message,
            "details": details or {}
        }
    } 