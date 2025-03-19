"""
HTTP client utilities for the Semantic Scholar API Server.
"""

import os
import logging
import httpx
import asyncio
import time
from typing import Dict, Optional, Tuple, Any

from ..config import Config, ErrorType, RateLimitConfig
from .errors import create_error_response

logger = logging.getLogger(__name__)

# Global HTTP client for connection pooling
http_client = None

class RateLimiter:
    """
    Rate limiter for API requests to prevent exceeding API limits.
    """
    def __init__(self):
        self._last_call_time = {}
        self._locks = {}

    def _get_rate_limit(self, endpoint: str) -> Tuple[int, int]:
        """Get the appropriate rate limit for an endpoint."""
        if any(restricted in endpoint for restricted in RateLimitConfig.RESTRICTED_ENDPOINTS):
            if "batch" in endpoint:
                return RateLimitConfig.BATCH_LIMIT
            if "search" in endpoint:
                return RateLimitConfig.SEARCH_LIMIT
            return RateLimitConfig.DEFAULT_LIMIT
        return RateLimitConfig.DEFAULT_LIMIT

    async def acquire(self, endpoint: str):
        """
        Acquire permission to make a request, waiting if necessary to respect rate limits.
        
        Args:
            endpoint: The API endpoint being accessed.
        """
        if endpoint not in self._locks:
            self._locks[endpoint] = asyncio.Lock()
            self._last_call_time[endpoint] = 0

        async with self._locks[endpoint]:
            rate_limit = self._get_rate_limit(endpoint)
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time[endpoint]
            
            if time_since_last_call < rate_limit[1]:
                delay = rate_limit[1] - time_since_last_call
                await asyncio.sleep(delay)
            
            self._last_call_time[endpoint] = time.time()

# Create global rate limiter instance
rate_limiter = RateLimiter()

def get_api_key() -> Optional[str]:
    """
    Get the Semantic Scholar API key from environment variables.
    Returns None if no API key is set, enabling unauthenticated access.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        logger.warning("No SEMANTIC_SCHOLAR_API_KEY set. Using unauthenticated access with lower rate limits.")
    return api_key

async def initialize_client():
    """Initialize the global HTTP client."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=Config.TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10)
        )
    return http_client

async def cleanup_client():
    """Clean up the global HTTP client."""
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None

async def make_request(endpoint: str, params: Dict = None) -> Dict:
    """
    Make a rate-limited request to the Semantic Scholar API.
    
    Args:
        endpoint: The API endpoint to call.
        params: Optional query parameters.
        
    Returns:
        The JSON response or an error response dictionary.
    """
    try:
        # Apply rate limiting
        await rate_limiter.acquire(endpoint)

        # Get API key if available
        api_key = get_api_key()
        headers = {"x-api-key": api_key} if api_key else {}
        url = f"{Config.BASE_URL}{endpoint}"

        # Use global client
        client = await initialize_client()
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} for {endpoint}: {e.response.text}")
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded. Consider using an API key for higher limits.",
                {
                    "retry_after": e.response.headers.get("retry-after"),
                    "authenticated": bool(get_api_key())
                }
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text}
        )
    except httpx.TimeoutException as e:
        logger.error(f"Request timeout for {endpoint}: {str(e)}")
        return create_error_response(
            ErrorType.TIMEOUT,
            f"Request timed out after {Config.TIMEOUT} seconds"
        )
    except Exception as e:
        logger.error(f"Unexpected error for {endpoint}: {str(e)}")
        return create_error_response(
            ErrorType.API_ERROR,
            str(e)
        ) 