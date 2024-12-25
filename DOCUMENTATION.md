# Semantic Scholar MCP Server Documentation

This document provides detailed documentation for all tools available in the Semantic Scholar MCP Server.

## Server Overview

The Semantic Scholar MCP Server is built using FastMCP and provides a comprehensive interface to the Semantic Scholar API. It enables LLMs to perform academic literature searches, retrieve paper details, analyze citation networks, and identify field experts.

## Base Configuration

- **Base URL**: `https://api.semanticscholar.org/graph/v1`
- **Authentication**: Requires Semantic Scholar API key via `SEMANTIC_SCHOLAR_API_KEY` environment variable
- **Request Headers**: Uses `x-api-key` header for API authentication

## Available Tools

### 1. custom_search_papers_semantic_scholar

**Purpose**: Performs a customizable search for academic papers on Semantic Scholar.

**Function Signature**:

```python
async def custom_search_papers_semantic_scholar(
    context: Context,
    query: str,
    limit: int = 10,
    fields: Optional[List[str]] = None
) -> Dict
```

**Parameters**:

- `query` (str): The search query string
- `limit` (int, optional): Maximum number of results (default: 10, max: 100)
- `fields` (List[str], optional): Fields to include in results

**Default Fields**:

- title
- abstract
- year
- citationCount
- authors
- url

**Returns**: Dictionary containing search results or error message

**Example Usage**:

```python
results = await custom_search_papers_semantic_scholar(
    context,
    query="machine learning",
    limit=5,
    fields=["title", "abstract", "year"]
)
```

### 2. get_paper_details_semantic_scholar

**Purpose**: Retrieves comprehensive information about a specific paper.

**Function Signature**:

```python
async def get_paper_details_semantic_scholar(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

**Parameters**:

- `paper_id` (str): Semantic Scholar paper ID
- `fields` (List[str], optional): Fields to include in results

**Default Fields**:

- title
- abstract
- year
- citationCount
- authors
- references
- citations
- url
- venue

**Returns**: Dictionary containing paper details or error message

### 3. get_author_details_semantic_scholar

**Purpose**: Retrieves detailed information about an academic author.

**Function Signature**:

```python
async def get_author_details_semantic_scholar(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None
) -> Dict
```

**Parameters**:

- `author_id` (str): Semantic Scholar author ID
- `fields` (List[str], optional): Fields to include in results

**Default Fields**:

- name
- paperCount
- citationCount
- hIndex
- papers

**Returns**: Dictionary containing author details or error message

### 4. advanced_search_papers_semantic_scholar

**Purpose**: Provides advanced search functionality with comprehensive filtering options for finding both recent and influential papers.

**Function Signature**:

```python
async def advanced_search_papers_semantic_scholar(
    context: Context,
    query: str,
    filters: Dict = {
        "year_range": None,  # (start, end)
        "min_citations": None,
        "sort_by": "relevance",  # or "citations", "year", "influence"
        "limit": 10
    }
) -> List[Dict]
```

**Parameters**:

- `query` (str): Search query for finding relevant papers
- `filters` (Dict): Dictionary containing search filters:
  - `year_range` (Tuple[int, int], optional): Tuple of (start_year, end_year) to filter papers by publication date
  - `min_citations` (int, optional): Minimum number of citations required
  - `sort_by` (str): Sorting criterion (default: "relevance")
    - "relevance": Sort by relevance to query
    - "citations": Sort by citation count
    - "year": Sort by publication year
    - "influence": Sort by influence score
  - `limit` (int): Maximum number of papers to return (default: 10)

**Returns**: List of dictionaries containing papers matching the search criteria

**Example Usage**:

```python
# Find recent influential papers in machine learning
results = await advanced_search_papers_semantic_scholar(
    context,
    query="machine learning",
    filters={
        "year_range": (2020, 2024),
        "min_citations": 50,
        "sort_by": "citations",
        "limit": 5
    }
)

# Find seminal papers in a field
results = await advanced_search_papers_semantic_scholar(
    context,
    query="quantum computing",
    filters={
        "min_citations": 1000,
        "sort_by": "influence",
        "limit": 10
    }
)
```

## Error Handling

All tools include comprehensive error handling:

1. **API Errors**:

   - HTTP status errors
   - Rate limiting
   - Authentication failures

2. **Input Validation**:

   - Parameter type checking
   - Value range validation
   - Required field verification

3. **Response Processing**:
   - JSON parsing errors
   - Missing field handling
   - Empty result handling

## Best Practices

1. **Rate Limiting**:

   - Respect API rate limits
   - Implement exponential backoff for retries

2. **Field Selection**:

   - Only request needed fields
   - Use default fields when unsure

3. **Result Limits**:

   - Use appropriate search limits
   - Consider pagination for large results

4. **Error Handling**:
   - Always check for error responses
   - Log errors appropriately
   - Provide meaningful error messages
