# Semantic Scholar Server Documentation

## Overview

The Semantic Scholar Server provides a comprehensive interface to the Semantic Scholar Academic Graph API through FastMCP. It offers tools for paper search, citation analysis, author information, and paper recommendations.

## Server Architecture

### Core Components

1. **Rate Limiting**

   - Configurable rate limits for different endpoints
   - Automatic adjustment based on authentication status
   - Queue-based request management

2. **Error Handling**

   - Standardized error responses
   - Comprehensive error types
   - Detailed error context

3. **Resource Management**

   - Connection pooling
   - Graceful shutdown
   - Resource cleanup

4. **Field Management**
   - Predefined field combinations
   - Field validation
   - Nested field support

## Available Tools

### Paper Search

#### 1. `paper_search`

Basic paper search with comprehensive filtering options.

```python
async def paper_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = Config.DEFAULT_PAGE_SIZE
) -> Dict
```

#### 2. `paper_search_match`

Exact title matching with filtering.

```python
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
) -> Dict
```

### Paper Details

#### 1. `paper_details`

Get detailed information about a single paper.

```python
async def paper_details(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

#### 2. `paper_batch_details`

Get details for multiple papers in one request.

```python
async def paper_batch_details(
    context: Context,
    paper_ids: List[str],
    fields: Optional[str] = None
) -> Dict
```

### Citations and References

#### 1. `paper_citations`

Get papers that cite the specified paper.

```python
async def paper_citations(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 2. `paper_references`

Get papers cited by the specified paper.

```python
async def paper_references(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

### Author Information

#### 1. `author_search`

Search for authors by name.

```python
async def author_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 2. `author_details`

Get detailed information about an author.

```python
async def author_details(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

#### 3. `author_papers`

Get papers written by an author.

```python
async def author_papers(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 4. `author_batch_details`

Get details for multiple authors in one request.

```python
async def author_batch_details(
    context: Context,
    author_ids: List[str],
    fields: Optional[str] = None
) -> Dict
```

### Paper Recommendations

#### `get_paper_recommendations`

Get paper recommendations with support for both single and multi-paper recommendations.

```python
async def get_paper_recommendations(
    context: Context,
    positive_paper_ids: List[str],
    negative_paper_ids: Optional[List[str]] = None,
    fields: Optional[str] = None,
    limit: int = 100,
    from_pool: str = "recent"
) -> Dict
```

**Features:**

- Single-paper recommendations with pool selection
- Multi-paper recommendations with positive/negative examples
- Automatic endpoint selection based on input
- Support for field customization

### Advanced Search

#### `advanced_search_papers_semantic_scholar`

Complex paper search with ranking strategies and advanced filtering.

```python
async def advanced_search_papers_semantic_scholar(
    context: Context,
    query: str,
    search_type: str = "comprehensive",
    filters: Optional[Dict] = None,
    search_config: Optional[Dict] = None
) -> Dict
```

**Search Types:**

- `comprehensive`: Balanced search
- `influential`: Focus on highly-cited papers
- `latest`: Focus on recent papers

## Field Constants

### Paper Fields

```python
class PaperFields:
    DEFAULT = ["title", "abstract", "year", "citationCount", "authors", "url"]
    DETAILED = DEFAULT + ["references", "citations", "venue", "influentialCitationCount"]
    MINIMAL = ["title", "year", "authors"]
    SEARCH = ["paperId", "title", "year", "citationCount"]
```

### Author Fields

```python
class AuthorDetailFields:
    BASIC = ["name", "url", "affiliations"]
    PAPERS_BASIC = ["papers"]
    PAPERS_DETAILED = ["papers.year", "papers.authors", "papers.abstract", "papers.venue", "papers.url"]
    COMPLETE = BASIC + ["papers", "papers.year", "papers.authors", "papers.venue"]
    METRICS = ["citationCount", "hIndex", "paperCount"]
```

### Citation Fields

```python
class CitationReferenceFields:
    BASIC = ["title"]
    CONTEXT = ["contexts", "intents", "isInfluential"]
    DETAILED = ["title", "abstract", "authors", "year", "venue"]
    COMPLETE = CONTEXT + DETAILED
```

## Error Handling

### Error Types

```python
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
```

### Error Response Format

```python
{
    "error": {
        "type": "error_type",
        "message": "Error description",
        "details": {
            "authenticated": true/false,
            # Additional context specific to error type
        }
    }
}
```

## Rate Limiting

### Configuration

```python
class RateLimitConfig:
    SEARCH_LIMIT = (1, 1)    # 1 request per second
    BATCH_LIMIT = (1, 1)     # 1 request per second
    DEFAULT_LIMIT = (10, 1)  # 10 requests per second
```

### Restricted Endpoints

- `/paper/batch`
- `/paper/search`
- `/recommendations`

## Best Practices

1. **Field Selection**

   - Use predefined field combinations when possible
   - Request only needed fields to improve performance
   - Validate fields before making requests

2. **Rate Limiting**

   - Use API key for higher rate limits
   - Implement retry logic for rate limit errors
   - Group requests using batch endpoints

3. **Error Handling**

   - Check for specific error types
   - Handle rate limit errors appropriately
   - Provide meaningful error messages to users

4. **Resource Management**

   - Use connection pooling for better performance
   - Implement proper cleanup in finally blocks
   - Handle graceful shutdowns

5. **Search Optimization**

   - Use `paper_search_match` for exact title matches
   - Use `advanced_search` for complex queries
   - Leverage search types for specific use cases

6. **Batch Operations**
   - Use batch endpoints for multiple items
   - Stay within batch size limits
   - Handle partial success responses
