# Semantic Scholar MCP Server Documentation

This document provides detailed documentation for all tools available in the Semantic Scholar MCP Server.

## Server Overview

The Semantic Scholar MCP Server is built using FastMCP and provides a comprehensive interface to the Semantic Scholar API. It enables LLMs to perform academic literature searches, retrieve paper details, manage citations and references, and interact with author information.

## Base Configuration

- **Base URL**: `https://api.semanticscholar.org/graph/v1`
- **Authentication**: Requires Semantic Scholar API key via `SEMANTIC_SCHOLAR_API_KEY` environment variable
- **Rate Limiting**:
  - Search and Batch endpoints: 1 request per second
  - Other endpoints: 10 requests per second
- **Timeout**: 30 seconds for all requests

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

### Citation/Reference Fields

```python
class CitationReferenceFields:
    BASIC = ["title"]
    CONTEXT = ["contexts", "intents", "isInfluential"]
    DETAILED = ["title", "abstract", "authors", "year", "venue"]
    COMPLETE = CONTEXT + DETAILED
```

## Available Tools

### Paper Search and Details

#### 1. paper_search

**Purpose**: Search for papers using various filters and criteria.

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
    limit: int = 10
) -> Dict
```

**Example**:

```python
# Search for recent machine learning papers
results = await paper_search(
    context,
    query="machine learning",
    year="2020-2024",
    min_citation_count=50,
    fields=["title", "abstract", "authors"]
)
```

#### 2. paper_search_match

**Purpose**: Find a specific paper by title match.

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

#### 3. paper_details

**Purpose**: Get detailed information about a specific paper.

```python
async def paper_details(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

#### 4. paper_batch_details

**Purpose**: Get details for multiple papers in a single request.

```python
async def paper_batch_details(
    context: Context,
    paper_ids: List[str],
    fields: Optional[str] = None
) -> Dict
```

### Citations and References

#### 5. paper_citations

**Purpose**: Get papers that cite a specific paper.

```python
async def paper_citations(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 6. paper_references

**Purpose**: Get papers cited by a specific paper.

```python
async def paper_references(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

### Author Tools

#### 7. author_search

**Purpose**: Search for authors by name.

```python
async def author_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 8. author_details

**Purpose**: Get detailed information about an author.

```python
async def author_details(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

#### 9. author_papers

**Purpose**: Get papers written by an author.

```python
async def author_papers(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100
) -> Dict
```

#### 10. author_batch_details

**Purpose**: Get details for multiple authors in a single request.

```python
async def author_batch_details(
    context: Context,
    author_ids: List[str],
    fields: Optional[str] = None
) -> Dict
```

### Advanced Search and Recommendations

#### 11. advanced_search_papers_semantic_scholar

**Purpose**: Advanced paper search with complex filtering and ranking.

```python
async def advanced_search_papers_semantic_scholar(
    context: Context,
    query: str,
    search_type: str = "comprehensive",
    filters: Optional[Dict] = None,
    search_config: Optional[Dict] = None
) -> Dict
```

**Search Types**:

- `comprehensive`: Balanced search considering relevance and impact
- `influential`: Focus on highly-cited papers
- `latest`: Focus on recent papers

#### 12. get_paper_recommendations

**Purpose**: Get paper recommendations based on example papers.

```python
async def get_paper_recommendations(
    context: Context,
    positive_paper_ids: List[str],
    negative_paper_ids: Optional[List[str]] = None,
    from_pool: Optional[str] = "recent",
    fields: Optional[str] = None,
    limit: int = 100
) -> Dict
```

## Error Handling

The server implements standardized error handling through the `ErrorType` enum:

```python
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
```

Each error response includes:

- Error type
- Descriptive message
- Additional details when relevant

## Best Practices

1. **Rate Limiting**:

   - Respect the different rate limits for search/batch (1/s) and other endpoints (10/s)
   - Use batch endpoints when possible to optimize request usage

2. **Field Selection**:

   - Use predefined field combinations when appropriate (e.g., `PaperFields.DEFAULT`)
   - Only request needed fields to optimize response size
   - Validate fields against `VALID_FIELDS` sets

3. **Pagination**:

   - Use offset/limit parameters for large result sets
   - Default limit is 100, maximum is 1000 for most endpoints
   - Batch endpoints have specific limits (500 for papers, 1000 for authors)

4. **Search Optimization**:

   - Use `paper_search_match` for exact title matches
   - Use `advanced_search_papers_semantic_scholar` for complex queries
   - Use batch endpoints for multiple paper/author lookups

5. **Error Handling**:
   - Always check for error responses
   - Handle rate limiting with appropriate backoff
   - Validate inputs before making requests
