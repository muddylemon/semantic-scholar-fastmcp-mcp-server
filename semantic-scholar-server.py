#!/usr/bin/env python3
from fastmcp import FastMCP, Context
import httpx
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import asyncio
import time
import signal
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global HTTP client for connection pooling
http_client = None

# Rate Limiting Configuration
@dataclass
class RateLimitConfig:
    # Define rate limits (requests, seconds)
    SEARCH_LIMIT = (1, 1)  # 1 request per 1 second
    BATCH_LIMIT = (1, 1)   # 1 request per 1 second
    DEFAULT_LIMIT = (10, 1)  # 10 requests per 1 second
    
    # Endpoints categorization
    RESTRICTED_ENDPOINTS = [
        "/paper/batch",
        "/paper/search",
        "/recommendations"
    ]

# Error Types
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    TIMEOUT = "timeout"

# Field Constants
class PaperFields:
    DEFAULT = ["title", "abstract", "year", "citationCount", "authors", "url"]
    DETAILED = DEFAULT + ["references", "citations", "venue", "influentialCitationCount"]
    MINIMAL = ["title", "year", "authors"]
    SEARCH = ["paperId", "title", "year", "citationCount"]
    
    # Valid fields from API documentation
    VALID_FIELDS = {
        "abstract",
        "authors",
        "citationCount",
        "citations",
        "corpusId",
        "embedding",
        "externalIds",
        "fieldsOfStudy",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "paperId",
        "publicationDate",
        "publicationTypes",
        "publicationVenue",
        "references",
        "s2FieldsOfStudy",
        "title",
        "tldr",
        "url",
        "venue",
        "year"
    }

class AuthorDetailFields:
    """Common field combinations for author details"""
    
    # Basic author information
    BASIC = ["name", "url", "affiliations"]
    
    # Author's papers information
    PAPERS_BASIC = ["papers"]  # Returns paperId and title
    PAPERS_DETAILED = [
        "papers.year",
        "papers.authors",
        "papers.abstract",
        "papers.venue",
        "papers.url"
    ]
    
    # Complete author profile
    COMPLETE = BASIC + ["papers", "papers.year", "papers.authors", "papers.venue"]
    
    # Citation metrics
    METRICS = ["citationCount", "hIndex", "paperCount"]

    # Valid fields for author details
    VALID_FIELDS = {
        "authorId",
        "name",
        "url",
        "affiliations",
        "papers",
        "papers.year",
        "papers.authors",
        "papers.abstract",
        "papers.venue",
        "papers.url",
        "citationCount",
        "hIndex",
        "paperCount"
    }

class PaperDetailFields:
    """Common field combinations for paper details"""
    
    # Basic paper information
    BASIC = ["title", "abstract", "year", "venue"]
    
    # Author information
    AUTHOR_BASIC = ["authors"]
    AUTHOR_DETAILED = ["authors.url", "authors.paperCount", "authors.citationCount"]
    
    # Citation information
    CITATION_BASIC = ["citations", "references"]
    CITATION_DETAILED = ["citations.title", "citations.abstract", "citations.year",
                        "references.title", "references.abstract", "references.year"]
    
    # Full paper details
    COMPLETE = BASIC + AUTHOR_BASIC + CITATION_BASIC + ["url", "fieldsOfStudy", 
                                                       "publicationVenue", "publicationTypes"]

class CitationReferenceFields:
    """Common field combinations for citation and reference queries"""
    
    # Basic information
    BASIC = ["title"]
    
    # Citation/Reference context
    CONTEXT = ["contexts", "intents", "isInfluential"]
    
    # Paper details
    DETAILED = ["title", "abstract", "authors", "year", "venue"]
    
    # Full information
    COMPLETE = CONTEXT + DETAILED

    # Valid fields for citation/reference queries
    VALID_FIELDS = {
        "contexts",
        "intents",
        "isInfluential",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "paperId",
        "url",
        "citationCount",
        "influentialCitationCount"
    }

# Configuration
class Config:
    # API Configuration
    API_VERSION = "v1"
    BASE_URL = f"https://api.semanticscholar.org/graph/{API_VERSION}"
    TIMEOUT = 30  # seconds
    
    # Request Limits
    MAX_BATCH_SIZE = 100
    MAX_RESULTS_PER_PAGE = 100
    DEFAULT_PAGE_SIZE = 10
    MAX_BATCHES = 5
    
    # Fields Configuration
    DEFAULT_FIELDS = PaperFields.DEFAULT
    
    # Feature Flags
    ENABLE_CACHING = False
    DEBUG_MODE = False
    
    # Search Configuration
    SEARCH_TYPES = {
        "comprehensive": {
            "description": "Balanced search considering relevance and impact",
            "min_citations": None,
            "ranking_strategy": "balanced"
        },
        "influential": {
            "description": "Focus on highly-cited and influential papers",
            "min_citations": 50,
            "ranking_strategy": "citations"
        },
        "latest": {
            "description": "Focus on recent papers with impact",
            "min_citations": None,
            "ranking_strategy": "recency"
        }
    }

# Rate Limiter
class RateLimiter:
    def __init__(self):
        self._last_call_time = {}
        self._locks = {}

    def _get_rate_limit(self, endpoint: str) -> Tuple[int, int]:
        if any(restricted in endpoint for restricted in RateLimitConfig.RESTRICTED_ENDPOINTS):
            return RateLimitConfig.SEARCH_LIMIT
        return RateLimitConfig.DEFAULT_LIMIT

    async def acquire(self, endpoint: str):
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

def create_error_response(
    error_type: ErrorType,
    message: str,
    details: Optional[Dict] = None
) -> Dict:
    return {
        "error": {
            "type": error_type.value,
            "message": message,
            "details": details or {}
        }
    }

mcp = FastMCP("Semantic Scholar Server")
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

async def handle_exception(loop, context):
    """Global exception handler for the event loop."""
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}")
    asyncio.create_task(shutdown())

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
    """Cleanup the global HTTP client."""
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None

async def make_request(endpoint: str, params: Dict = None) -> Dict:
    """Make a rate-limited request to the Semantic Scholar API."""
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

@mcp.tool()
async def paper_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,  # supports formats like "2019", "2016-2020", "2010-", "-2015"
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = Config.DEFAULT_PAGE_SIZE
) -> Dict:
    """
    Search for papers on Semantic Scholar using various filters.

    Args:
        query (str): A plain-text search query string
        fields (Optional[List[str]]): List of fields to return (paperId and title are always returned)
        publication_types (Optional[List[str]]): Filter by publication types (Review, JournalArticle, etc.)
        open_access_pdf (bool): Filter to only include papers with public PDF
        min_citation_count (Optional[int]): Minimum number of citations required
        year (Optional[str]): Publication year filter ("2019", "2016-2020", "2010-", "-2015")
        venue (Optional[List[str]]): Filter by publication venues
        fields_of_study (Optional[List[str]]): Filter by fields of study
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum results to return (default: 10, max: 100)

    Returns:
        Dict: {
            "total": int,  # Approximate number of matching results
            "offset": int, # Starting position
            "next": int,   # Starting position of next batch (if available)
            "data": List[Dict]  # List of papers
        }
    """
    if not query.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Query string cannot be empty"
        )

    # Validate and prepare fields
    if fields is None:
        fields = PaperFields.DEFAULT
    else:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)}
            )

    # Validate and prepare parameters
    limit = min(limit, Config.MAX_RESULTS_PER_PAGE)
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "fields": ",".join(fields)
    }

    # Add optional filters
    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)
    if open_access_pdf:
        params["openAccessPdf"] = "true"
    if min_citation_count is not None:
        params["minCitationCount"] = min_citation_count
    if year:
        params["year"] = year
    if venue:
        params["venue"] = ",".join(venue)
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    return await make_request("/paper/search", params)

async def process_search_results(
    context: Context,
    initial_results: Dict,
    filters: Dict,
    config: Dict
) -> Dict:
    """Helper function to process and enhance search results."""
    try:
        papers = initial_results.get("data", [])
        total_found = initial_results.get("total", 0)
        processed_papers = []
        
        # Track statistics
        stats = {
            "total_found": total_found,
            "total_processed": len(papers),
            "avg_citations": 0,
            "year_distribution": {},
            "top_venues": [],
            "top_authors": []
        }
        
        # Process each paper
        author_paper_count = {}
        venue_papers = {}
        total_citations = 0
        
        for paper in papers:
            # Apply complex filters
            if not await should_include_paper(paper, filters, config):
                continue
                
            # Track statistics
            year = paper.get("year")
            if year:
                stats["year_distribution"][year] = stats["year_distribution"].get(year, 0) + 1
            
            citations = paper.get("citationCount", 0)
            total_citations += citations
            
            venue = paper.get("venue")
            if venue:
                venue_papers[venue] = venue_papers.get(venue, 0) + 1
            
            for author in paper.get("authors", []):
                author_name = author.get("name")
                if author_name:
                    author_paper_count[author_name] = author_paper_count.get(author_name, 0) + 1
            
            processed_papers.append(paper)
        
        # Calculate statistics
        if processed_papers:
            stats["avg_citations"] = total_citations / len(processed_papers)
        
        # Get top venues and authors
        stats["top_venues"] = sorted(venue_papers.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["top_authors"] = sorted(author_paper_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Apply ranking strategy
        processed_papers = await rank_papers(processed_papers, config["ranking_strategy"])
        
        # Apply diversification if enabled
        if config["diversify_results"]:
            processed_papers = await diversify_results(processed_papers)
        
        return {
            "papers": processed_papers,
            "stats": stats,
            "meta": {
                "search_coverage": len(processed_papers) / total_found if total_found > 0 else 0,
                "filters_applied": list(filters.keys()),
                "ranking_factors": await get_ranking_factors(config["ranking_strategy"])
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing search results: {str(e)}")
        return create_error_response(
            ErrorType.API_ERROR,
            f"Error processing search results: {str(e)}"
        )

async def should_include_paper(paper: Dict, filters: Dict, config: Dict) -> bool:
    """Determine if a paper should be included based on complex filters."""
    try:
        # Check citation range
        if filters["citation_range"]:
            min_cites, max_cites = filters["citation_range"]
            citations = paper.get("citationCount", 0)
            if not (min_cites <= citations <= max_cites):
                return False
        
        # Check abstract requirement and length
        if filters["require_abstract"] or config["min_abstract_length"]:
            abstract = paper.get("abstract", "")
            if not abstract:
                return False
            if config["min_abstract_length"] and len(abstract) < config["min_abstract_length"]:
                return False
        
        # Check references requirement
        if filters["require_references"] and not paper.get("references"):
            return False
        
        # Check venue filters
        venue = paper.get("venue", "")
        if filters["exclude_venues"] and venue in filters["exclude_venues"]:
            return False
        if filters["include_venues"] and venue not in filters["include_venues"]:
            return False
        
        # Check author criteria
        if filters["author_citation_count"]:
            for author in paper.get("authors", []):
                if author.get("citationCount", 0) >= filters["author_citation_count"]:
                    return True
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error in paper filtering: {str(e)}")
        return False

async def rank_papers(papers: List[Dict], strategy: str) -> List[Dict]:
    """Rank papers based on the specified strategy."""
    try:
        if strategy == "citations":
            return sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)
        elif strategy == "recency":
            return sorted(papers, key=lambda x: (x.get("year", 0), x.get("citationCount", 0)), reverse=True)
        elif strategy == "balanced":
            return sorted(papers, key=lambda x: (
                x.get("citationCount", 0) * 0.6 +
                x.get("year", 0) * 0.4
            ), reverse=True)
        return papers
    except Exception as e:
        logger.error(f"Error ranking papers: {str(e)}")
        return papers

async def diversify_results(papers: List[Dict]) -> List[Dict]:
    """Apply diversity to results to avoid similar papers."""
    # Implementation for result diversification
    # This could involve:
    # 1. Clustering papers by topic
    # 2. Ensuring representation from different venues
    # 3. Spreading papers across years
    # 4. Limiting papers per author
    return papers

async def get_ranking_factors(strategy: str) -> Dict:
    """Get the factors used in ranking for transparency."""
    if strategy == "citations":
        return {"citations": 1.0}
    elif strategy == "recency":
        return {"year": 0.7, "citations": 0.3}
    elif strategy == "balanced":
        return {"citations": 0.6, "year": 0.4}
    return {}

@mcp.tool()
async def paper_search_match(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None
) -> Dict:
    """
    Search for a single paper by title match on Semantic Scholar. Returns the closest matching paper.

    Args:
        query (str): Paper title to search for
        fields (Optional[List[str]]): List of fields to return (paperId and title are always returned)
        publication_types (Optional[List[str]]): Filter by publication types (Review, JournalArticle, etc.)
        open_access_pdf (bool): Filter to only include papers with public PDF
        min_citation_count (Optional[int]): Minimum number of citations required
        year (Optional[str]): Publication year filter ("2019", "2016-2020", "2010-", "-2015")
        venue (Optional[List[str]]): Filter by publication venues
        fields_of_study (Optional[List[str]]): Filter by fields of study

    Returns:
        Dict: The closest matching paper with requested fields and matchScore, 
              or error message if no match found
    """
    if not query.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Query string cannot be empty"
        )

    # Validate and prepare fields
    if fields is None:
        fields = PaperFields.DEFAULT
    else:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)}
            )

    # Build base parameters
    params = {"query": query}

    # Add optional parameters
    if fields:
        params["fields"] = ",".join(fields)
    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)
    if open_access_pdf:
        params["openAccessPdf"] = "true"
    if min_citation_count is not None:
        params["minCitationCount"] = str(min_citation_count)
    if year:
        params["year"] = year
    if venue:
        params["venue"] = ",".join(venue)
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    result = await make_request("/paper/search/match", params)
    
    # Handle specific error cases
    if isinstance(result, Dict):
        if "error" in result:
            error_msg = result["error"].get("message", "")
            if "404" in error_msg:
                return create_error_response(
                    ErrorType.VALIDATION,
                    "No matching paper found",
                    {"original_query": query}
                )
            return result
    
    return result

@mcp.tool()
async def paper_details(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None
) -> Dict:
    """
    Get detailed information about a paper using various types of identifiers.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
              Supported URLs from: semanticscholar.org, arxiv.org, aclweb.org,
                                 acm.org, biorxiv.org
        
        fields (Optional[List[str]]): List of fields to return. Special syntax:
            - For author fields: "author.url", "author.paperCount", etc.
            - For citation/reference fields: "citations.title", "references.abstract", etc.
            - For embeddings: "embedding.specter_v2" for v2 embeddings
            If omitted, returns only paperId and title.

    Returns:
        Dict: Paper details with requested fields. paperId is always included.
              Returns error message if paper not found or other issues occur.

    Examples:
        >>> # Get basic paper info
        >>> paper_details("DOI:10.18653/v1/N18-3011")
        
        >>> # Get specific fields including nested citation data
        >>> paper_details(
        ...     "ARXIV:2106.15928",
        ...     fields=["title", "authors", "citations.title", "citations.abstract"]
        ... )
        
        >>> # Get paper by direct URL
        >>> paper_details(
        ...     "URL:https://arxiv.org/abs/2106.15928v1",
        ...     fields=["url", "year", "authors"]
        ... )
    """
    if not paper_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Paper ID cannot be empty"
        )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}", params)
    
    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id}
            )
        return result

    return result

@mcp.tool()
async def paper_authors(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict:
    """
    Get details about a paper's authors with pagination support.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
              Supported URLs from: semanticscholar.org, arxiv.org, aclweb.org,
                                 acm.org, biorxiv.org
        
        fields (Optional[List[str]]): List of fields to return. Examples:
            - Basic fields: "name", "url", "affiliations"
            - Paper fields: "papers" returns papers with paperId and title
            - Nested paper fields: "papers.year", "papers.authors", "papers.abstract"
            If omitted, returns only authorId and name.
            
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum number of authors to return (default: 100, max: 1000)

    Returns:
        Dict: {
            "offset": int,    # Current offset
            "next": int,      # Next offset (if more results available)
            "data": List[Dict] # List of authors with requested fields
        }

    Examples:
        >>> # Get basic author information
        >>> paper_authors("DOI:10.18653/v1/N18-3011")
        
        >>> # Get authors with affiliations and papers, limited to first 2 authors
        >>> paper_authors(
        ...     "ARXIV:2106.15928",
        ...     fields=["affiliations", "papers"],
        ...     limit=2
        ... )
        
        >>> # Get last author with detailed paper information
        >>> paper_authors(
        ...     "649def34f8be52c8b66281af98ae884c09aef38b",
        ...     fields=["url", "papers.year", "papers.authors"],
        ...     offset=2
        ... )
    """
    if not paper_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Paper ID cannot be empty"
        )

    # Validate limit
    if limit > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Limit cannot exceed 1000",
            {"max_limit": 1000}
        )
    
    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {
        "offset": offset,
        "limit": limit
    }
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/authors", params)
    
    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id}
            )
        return result

    return result

@mcp.tool()
async def paper_citations(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict:
    """
    Get papers that cite the specified paper (papers where this paper appears in their bibliography).

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id>, ACL:<id>, PMID:<id>, PMCID:<id>
            - URL:<url> (from semanticscholar.org, arxiv.org, aclweb.org, acm.org, biorxiv.org)
        
        fields (Optional[List[str]]): List of fields to return. Examples:
            - Citation metadata: "contexts", "intents", "isInfluential"
            - Paper fields: "title", "abstract", "authors"
            If omitted, returns only paperId and title.
        
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum results to return (default: 100, max: 1000)

    Returns:
        Dict: {
            "offset": int,    # Current offset
            "next": int,      # Next offset (if more results available)
            "data": List[Dict] # List of citations with requested fields
        }

    Examples:
        >>> # Get basic citation information
        >>> paper_citations("DOI:10.18653/v1/N18-3011")
        
        >>> # Get detailed citation context with a specific offset
        >>> paper_citations(
        ...     "ARXIV:2106.15928",
        ...     fields=["contexts", "intents", "isInfluential", "abstract"],
        ...     offset=200,
        ...     limit=10
        ... )
    """
    if not paper_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Paper ID cannot be empty"
        )

    # Validate limit
    if limit > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Limit cannot exceed 1000",
            {"max_limit": 1000}
        )

    # Validate fields
    if fields:
        invalid_fields = set(fields) - CitationReferenceFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(CitationReferenceFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {
        "offset": offset,
        "limit": limit
    }
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/citations", params)
    
    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id}
            )
        return result

    return result

@mcp.tool()
async def paper_references(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict:
    """
    Get papers cited by the specified paper (papers appearing in this paper's bibliography).

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id>, ACL:<id>, PMID:<id>, PMCID:<id>
            - URL:<url> (from semanticscholar.org, arxiv.org, aclweb.org, acm.org, biorxiv.org)
        
        fields (Optional[List[str]]): List of fields to return. Examples:
            - Reference metadata: "contexts", "intents", "isInfluential"
            - Paper fields: "title", "abstract", "authors"
            If omitted, returns only paperId and title.
        
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum results to return (default: 100, max: 1000)

    Returns:
        Dict: {
            "offset": int,    # Current offset
            "next": int,      # Next offset (if more results available)
            "data": List[Dict] # List of references with requested fields
        }

    Examples:
        >>> # Get basic reference information
        >>> paper_references("DOI:10.18653/v1/N18-3011")
        
        >>> # Get detailed reference information
        >>> paper_references(
        ...     "ARXIV:2106.15928",
        ...     fields=["contexts", "intents", "isInfluential", "abstract"],
        ...     offset=200,
        ...     limit=10
        ... )
    """
    if not paper_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Paper ID cannot be empty"
        )

    # Validate limit
    if limit > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Limit cannot exceed 1000",
            {"max_limit": 1000}
        )

    # Validate fields
    if fields:
        invalid_fields = set(fields) - CitationReferenceFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(CitationReferenceFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {
        "offset": offset,
        "limit": limit
    }
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/references", params)
    
    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id}
            )
        return result

    return result

@mcp.tool()
async def author_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict:
    """
    Search for authors by name on Semantic Scholar.
    
    Args:
        query (str): A plain-text search query string for author names
        fields (Optional[List[str]]): List of fields to return. Examples:
            - Basic author info: "name", "url", "affiliations"
            - Paper info: "papers.title", "papers.year", "papers.authors"
            If omitted, returns only authorId and name.
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum results to return (default: 100, max: 1000)
    
    Returns:
        Dict: {
            "total": int,    # Total number of matching authors
            "offset": int,   # Current offset
            "next": int,     # Next offset (if more results available)
            "data": List[Dict] # List of authors with requested fields
        }
    """
    if not query.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Query string cannot be empty"
        )

    # Validate limit
    if limit > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Limit cannot exceed 1000",
            {"max_limit": 1000}
        )

    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {
        "query": query,
        "offset": offset,
        "limit": limit
    }
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    return await make_request("/author/search", params)

@mcp.tool()
async def author_details(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None
) -> Dict:
    """
    Get detailed information about an author.
    
    Args:
        author_id (str): Semantic Scholar author ID
        fields (Optional[List[str]]): List of fields to return. Examples:
            - Basic info: "name", "url", "affiliations"
            - Papers: "papers", "papers.year", "papers.authors"
            - Metrics: "citationCount", "hIndex", "paperCount"
            If omitted, returns only authorId and name.
    
    Returns:
        Dict: Author details with requested fields
    """
    if not author_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Author ID cannot be empty"
        )

    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/author/{author_id}", params)
    
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Author not found",
                {"author_id": author_id}
            )
        return result

    return result

@mcp.tool()
async def author_papers(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict:
    """
    Get papers written by an author with pagination support.
    
    Args:
        author_id (str): Semantic Scholar author ID
        fields (Optional[List[str]]): List of fields to return for each paper. Examples:
            - Basic paper info: "title", "abstract", "year"
            - Citations: "citations", "citations.title"
            - Authors: "authors", "authors.name"
            If omitted, returns only paperId and title for each paper.
        offset (int): Pagination offset (default: 0)
        limit (int): Maximum papers to return (default: 100, max: 1000)
    
    Returns:
        Dict: {
            "offset": int,   # Current offset 
            "next": int,     # Next offset (if more results available)
            "data": List[Dict] # List of papers with requested fields
        }
    """
    if not author_id.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Author ID cannot be empty"
        )

    # Validate limit
    if limit > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Limit cannot exceed 1000",
            {"max_limit": 1000}
        )

    # Validate fields
    if fields:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {
        "offset": offset,
        "limit": limit
    }
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/author/{author_id}/papers", params)
    
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(
                ErrorType.VALIDATION,
                "Author not found",
                {"author_id": author_id}
            )
        return result

    return result

@mcp.tool()
async def paper_batch_details(
    context: Context,
    paper_ids: List[str],
    fields: Optional[str] = None
) -> Dict:
    """
    Get details for multiple papers in a single batch request.
    
    Args:
        paper_ids (List[str]): List of paper identifiers in any supported format:
            - Semantic Scholar IDs
            - DOI: prefixed with "DOI:"
            - arXiv: prefixed with "ARXIV:"
            - MAG: prefixed with "MAG:"
            - ACL: prefixed with "ACL:"
            - PubMed: prefixed with "PMID:"
            - PubMed Central: prefixed with "PMCID:"
            - URL: prefixed with "URL:" (from supported domains)
        fields (Optional[str]): Comma-separated list of fields to return.
            Special syntax for nested fields:
            - Author fields: "authors.url", "authors.paperCount"
            - Citation/reference fields: "citations.title", "references.year"
            - Embedding: "embedding.specter_v2" for v2 embeddings
            If omitted, returns only paperId and title.
    
    Returns:
        List[Dict]: List of papers with requested fields. Papers maintain order
                   of input IDs. Invalid IDs return null in results.
                   
    Example:
        >>> # Get basic info for multiple papers
        >>> paper_batch_details(
        ...     paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", 
        ...               "ARXIV:2106.15928"],
        ...     fields="title,citationCount,authors"
        ... )
    """
    # Validate inputs
    if not paper_ids:
        return create_error_response(
            ErrorType.VALIDATION,
            "Paper IDs list cannot be empty"
        )
        
    if len(paper_ids) > 500:
        return create_error_response(
            ErrorType.VALIDATION,
            "Cannot process more than 500 paper IDs at once",
            {"max_papers": 500, "received": len(paper_ids)}
        )

    # Validate fields if provided
    if fields:
        field_list = fields.split(",")
        invalid_fields = set(field_list) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = fields

    # Make POST request with proper structure
    try:
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}
            
            response = await client.post(
                f"{Config.BASE_URL}/paper/batch",
                params=params,
                json={"ids": paper_ids},
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded",
                {"retry_after": e.response.headers.get("retry-after")}
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text}
        )
    except httpx.TimeoutException:
        return create_error_response(
            ErrorType.TIMEOUT,
            f"Request timed out after {Config.TIMEOUT} seconds"
        )
    except Exception as e:
        return create_error_response(
            ErrorType.API_ERROR,
            str(e)
        )

@mcp.tool()
async def author_batch_details(
    context: Context,
    author_ids: List[str],
    fields: Optional[str] = None
) -> Dict:
    """
    Get details for multiple authors in a single batch request.
    
    Args:
        author_ids (List[str]): List of Semantic Scholar author IDs
        fields (Optional[str]): Comma-separated list of fields to return.
            Special syntax for paper fields:
            - Basic fields: "name", "url", "affiliations"
            - Paper fields: "papers.title", "papers.year", "papers.authors"
            - Metrics: "citationCount", "hIndex", "paperCount"
            If omitted, returns only authorId and name.
    
    Returns:
        List[Dict]: List of authors with requested fields. Authors maintain order
                   of input IDs. Invalid IDs return null in results.
                   
    Example:
        >>> # Get citation metrics for multiple authors
        >>> author_batch_details(
        ...     author_ids=["1741101", "1780531"],
        ...     fields="name,hIndex,citationCount,paperCount"
        ... )
    """
    # Validate inputs
    if not author_ids:
        return create_error_response(
            ErrorType.VALIDATION,
            "Author IDs list cannot be empty"
        )
        
    if len(author_ids) > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Cannot process more than 1000 author IDs at once",
            {"max_authors": 1000, "received": len(author_ids)}
        )

    # Validate fields if provided
    if fields:
        field_list = fields.split(",")
        invalid_fields = set(field_list) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)}
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = fields

    # Make POST request with proper structure
    try:
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}
            
            response = await client.post(
                f"{Config.BASE_URL}/author/batch",
                params=params,
                json={"ids": author_ids},
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded",
                {"retry_after": e.response.headers.get("retry-after")}
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text}
        )
    except httpx.TimeoutException:
        return create_error_response(
            ErrorType.TIMEOUT,
            f"Request timed out after {Config.TIMEOUT} seconds"
        )
    except Exception as e:
        return create_error_response(
            ErrorType.API_ERROR,
            str(e)
        )

@mcp.tool()
async def get_paper_recommendations(
    context: Context,
    positive_paper_ids: List[str],
    negative_paper_ids: Optional[List[str]] = None,
    fields: Optional[str] = None,
    limit: int = 100,
    from_pool: str = "recent"
) -> Dict:
    """
    Get paper recommendations based on positive and optional negative example papers.
    Supports both single-paper and multi-paper recommendation endpoints.
    
    Args:
        positive_paper_ids (List[str]): List of paper IDs to use as positive examples.
            If only one paper ID is provided, uses the single-paper recommendation endpoint.
            Supports various ID formats:
            - Semantic Scholar ID
            - DOI: prefixed with "DOI:"
            - arXiv: prefixed with "ARXIV:"
            - MAG: prefixed with "MAG:"
            - ACL: prefixed with "ACL:"
            - PMID: prefixed with "PMID:"
            - PMCID: prefixed with "PMCID:"
            - URL: prefixed with "URL:" (from supported domains)
        negative_paper_ids (Optional[List[str]]): List of paper IDs to use as negative examples.
            Only used when using the multi-paper recommendation endpoint.
        fields (Optional[str]): Comma-separated list of fields to return for recommended papers.
            Examples: "title,url,authors", "abstract,year,venue"
            If omitted, returns only paperId and title.
        limit (int): Number of recommendations to return (default: 100, max: 500)
        from_pool (str): Which pool of papers to recommend from (default: "recent").
            Only used for single-paper recommendations.
            Options: "recent" (default), "all-cs"
    
    Returns:
        Dict: Contains list of recommended papers with requested fields
    
    Examples:
        >>> # Single paper recommendation
        >>> get_paper_recommendations(
        ...     positive_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b"],
        ...     fields="title,authors,year",
        ...     limit=10
        ... )
        
        >>> # Multi-paper recommendation with positive and negative examples
        >>> get_paper_recommendations(
        ...     positive_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", 
        ...                        "ARXIV:2106.15928"],
        ...     negative_paper_ids=["ArXiv:1805.02262"],
        ...     fields="title,abstract,authors"
        ... )
    """
    # Validate inputs
    if not positive_paper_ids:
        return create_error_response(
            ErrorType.VALIDATION,
            "Must provide at least one positive paper ID"
        )
    
    if limit > 500:
        return create_error_response(
            ErrorType.VALIDATION,
            "Cannot request more than 500 recommendations",
            {"max_limit": 500, "requested": limit}
        )
    
    if from_pool not in ["recent", "all-cs"]:
        return create_error_response(
            ErrorType.VALIDATION,
            "Invalid paper pool specified",
            {"valid_pools": ["recent", "all-cs"]}
        )

    # Validate fields if provided
    if fields:
        field_list = fields.split(",")
        invalid_fields = set(field_list) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)}
            )

    try:
        # Apply rate limiting through our standard mechanism
        endpoint = "/recommendations"
        await rate_limiter.acquire(endpoint)

        async with httpx.AsyncClient(timeout=Config.TIMEOUT, follow_redirects=True) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}
            
            # Choose endpoint based on number of input papers
            if len(positive_paper_ids) == 1 and not negative_paper_ids:
                # Single paper recommendation
                paper_id = positive_paper_ids[0]
                params = {
                    "limit": limit,
                    "from": from_pool
                }
                if fields:
                    params["fields"] = fields
                    
                url = f"https://api.semanticscholar.org/recommendations/v1/papers/{paper_id}"
                response = await client.get(url, params=params, headers=headers)
            else:
                # Multi-paper recommendation
                params = {"limit": limit}
                if fields:
                    params["fields"] = fields
                    
                request_body = {
                    "positivePaperIds": positive_paper_ids,
                    "negativePaperIds": negative_paper_ids or []
                }
                
                url = "https://api.semanticscholar.org/recommendations/v1/papers"
                response = await client.post(url, params=params, json=request_body, headers=headers)
            
            # Handle specific error cases
            if response.status_code == 404:
                return create_error_response(
                    ErrorType.VALIDATION,
                    "One or more input papers not found",
                    {
                        "status_code": 404,
                        "paper_ids": positive_paper_ids,
                        "details": "Please verify all paper IDs are valid"
                    }
                )
            elif response.status_code == 405:
                return create_error_response(
                    ErrorType.API_ERROR,
                    "Method not allowed - API endpoint may have changed",
                    {
                        "status_code": 405,
                        "endpoint": url
                    }
                )
            
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
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
            f"HTTP error {e.response.status_code}",
            {"response": e.response.text}
        )
    except httpx.TimeoutException as e:
        return create_error_response(
            ErrorType.TIMEOUT,
            f"Request timed out after {Config.TIMEOUT} seconds",
            {"timeout": Config.TIMEOUT}
        )
    except Exception as e:
        logger.error(f"Unexpected error in recommendations: {str(e)}")
        return create_error_response(
            ErrorType.API_ERROR,
            "Failed to get recommendations",
            {"error": str(e)}
        )

@mcp.tool()
async def advanced_search_papers_semantic_scholar(
    context: Context,
    query: str,
    search_type: str = "comprehensive",  # "comprehensive", "influential", "latest"
    filters: Optional[Dict] = None,
    search_config: Optional[Dict] = None
) -> Dict:
    """
    Advanced paper search with complex filtering and ranking strategies.
    This tool builds on top of the basic paper_search to provide additional 
    functionality like multi-criteria search and specialized result ranking.

    Args:
        query (str): Search query string
        search_type (str): Type of search to perform:
            - "comprehensive": Balanced search considering relevance and impact
            - "influential": Focus on highly-cited and influential papers
            - "latest": Focus on recent papers with impact
        filters (Optional[Dict]): Advanced filtering options:
            {
                "year_range": Optional[Tuple[int, int]],  # (start_year, end_year)
                "min_citations": Optional[int],
                "max_papers_per_author": Optional[int],
                "require_abstract": bool = False,
                "require_references": bool = False,
                "citation_range": Optional[Tuple[int, int]],  # (min, max)
                "author_citation_count": Optional[int],  # min author citations
                "venue_impact_factor": Optional[float],  # min venue impact
                "exclude_venues": Optional[List[str]],
                "include_venues": Optional[List[str]]
            }
        search_config (Optional[Dict]): Search behavior configuration:
            {
                "batch_size": int = 100,  # papers per batch
                "max_batches": int = 5,   # max number of batches to process
                "diversify_results": bool = True,  # avoid similar papers
                "prioritize_open_access": bool = False,
                "min_abstract_length": Optional[int],
                "ranking_strategy": str = "balanced"  # "citations", "recency", "relevance"
            }

    Returns:
        Dict: {
            "papers": List[Dict],  # Processed and ranked papers
            "stats": {
                "total_found": int,
                "total_processed": int,
                "avg_citations": float,
                "year_distribution": Dict[int, int],
                "top_venues": List[str],
                "top_authors": List[str]
            },
            "meta": {
                "search_coverage": float,  # % of total results processed
                "filters_applied": List[str],
                "ranking_factors": Dict[str, float]
            }
        }
    """
    if not query.strip():
        return create_error_response(
            ErrorType.VALIDATION,
            "Query string cannot be empty"
        )

    # Validate search type
    if search_type not in Config.SEARCH_TYPES:
        return create_error_response(
            ErrorType.VALIDATION,
            f"Invalid search type. Must be one of: {list(Config.SEARCH_TYPES.keys())}"
        )

    # Set default configurations
    default_filters = {
        "year_range": None,
        "min_citations": None,
        "max_papers_per_author": None,
        "require_abstract": False,
        "require_references": False,
        "citation_range": None,
        "author_citation_count": None,
        "venue_impact_factor": None,
        "exclude_venues": [],
        "include_venues": []
    }
    
    default_config = {
        "batch_size": Config.MAX_BATCH_SIZE,
        "max_batches": Config.MAX_BATCHES,
        "diversify_results": True,
        "prioritize_open_access": False,
        "min_abstract_length": None,
        "ranking_strategy": Config.SEARCH_TYPES[search_type]["ranking_strategy"]
    }

    # Merge provided filters/config with defaults
    filters = {**default_filters, **(filters or {})}
    search_config = {**default_config, **(search_config or {})}

    # Apply search type configurations
    search_type_config = Config.SEARCH_TYPES[search_type]
    if search_type_config["min_citations"]:
        filters["min_citations"] = max(
            filters.get("min_citations", 0),
            search_type_config["min_citations"]
        )
    
    if search_type == "latest":
        current_year = datetime.now().year
        if not filters.get("year_range"):
            filters["year_range"] = (current_year - 2, current_year)

    try:
        # Perform initial search using paper_search
        initial_results = await paper_search(
            context,
            query=query,
            fields=PaperFields.DETAILED,
            min_citation_count=filters.get("min_citations"),
            year=f"{filters['year_range'][0]}-{filters['year_range'][1]}" if filters["year_range"] else None,
            limit=search_config["batch_size"]
        )

        if "error" in initial_results:
            return initial_results

        # Process and enhance results
        processed_results = await process_search_results(
            context, initial_results, filters, search_config
        )

        return processed_results

    except Exception as e:
        logger.error(f"Error in advanced search: {str(e)}")
        return create_error_response(
            ErrorType.API_ERROR,
            str(e)
        )

async def shutdown():
    """Gracefully shut down the server."""
    logger.info("Initiating graceful shutdown...")
    
    # Cancel all tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Cleanup resources
    await cleanup_client()
    await mcp.cleanup()
    
    logger.info(f"Cancelled {len(tasks)} tasks")
    logger.info("Shutdown complete")

def init_signal_handlers(loop):
    """Initialize signal handlers for graceful shutdown."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    logger.info("Signal handlers initialized")

async def run_server():
    """Run the server with proper async context management."""
    async with mcp:
        try:
            # Initialize HTTP client
            await initialize_client()
            
            # Start the server
            logger.info("Starting Semantic Scholar Server")
            await mcp.run_async()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await shutdown()

if __name__ == "__main__":
    try:
        # Set up event loop with exception handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(handle_exception)
        
        # Initialize signal handlers
        init_signal_handlers(loop)
        
        # Run the server
        loop.run_until_complete(run_server())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        try:
            loop.run_until_complete(asyncio.sleep(0))  # Let pending tasks complete
            loop.close()
        except Exception as e:
            logger.error(f"Error during final cleanup: {str(e)}")
        logger.info("Server stopped")
