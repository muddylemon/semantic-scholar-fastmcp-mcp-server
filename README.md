# Semantic Scholar MCP Server

A FastMCP server implementation for the Semantic Scholar API, providing comprehensive access to academic paper data, author information, and citation networks.

## Features

- **Paper Search & Discovery**

  - Full-text search with advanced filtering
  - Title-based paper matching
  - Paper recommendations (single and multi-paper)
  - Batch paper details retrieval
  - Advanced search with ranking strategies

- **Citation Analysis**

  - Citation network exploration
  - Reference tracking
  - Citation context and influence analysis

- **Author Information**

  - Author search and profile details
  - Publication history
  - Batch author details retrieval

- **Advanced Features**
  - Complex search with multiple ranking strategies
  - Customizable field selection
  - Efficient batch operations
  - Rate limiting compliance
  - Support for both authenticated and unauthenticated access
  - Graceful shutdown and error handling
  - Connection pooling and resource management

## System Requirements

- Python 3.8+
- FastMCP framework
- `httpx` for async HTTP requests
- Environment variable for API key (optional)

## Installation

Install using FastMCP:

```bash
fastmcp install semantic-scholar-server.py --name "Semantic Scholar" -e SEMANTIC_SCHOLAR_API_KEY=your-api-key
```

The `-e SEMANTIC_SCHOLAR_API_KEY` parameter is optional. If not provided, the server will use unauthenticated access with lower rate limits.

## Configuration

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: Your Semantic Scholar API key (optional)
  - Get your key from [Semantic Scholar API](https://www.semanticscholar.org/product/api)
  - If not provided, the server will use unauthenticated access

### API Access Modes

#### Authenticated Access (With API Key)

- Higher rate limits
- Faster response times
- Access to all API features
- Recommended for production use

#### Unauthenticated Access (No API Key)

- Lower rate limits
- Longer timeouts
- Basic API functionality
- Suitable for testing and development

### Rate Limits

The server automatically adjusts to the appropriate rate limits:

**With API Key**:

- Search and batch endpoints: 1 request per second
- Other endpoints: 10 requests per second

**Without API Key**:

- All endpoints: 100 requests per 5 minutes
- Longer timeouts for requests

## Available MCP Tools

### Paper Search Tools

- `paper_search`: General paper search with comprehensive filters
- `paper_search_match`: Exact title matching
- `paper_details`: Single paper details
- `paper_batch_details`: Multiple paper details
- `advanced_search_papers_semantic_scholar`: Complex search with ranking strategies

### Citation Tools

- `paper_citations`: Get citing papers with context
- `paper_references`: Get referenced papers with context

### Author Tools

- `author_search`: Search for authors
- `author_details`: Get author information
- `author_papers`: Get author's publications
- `author_batch_details`: Multiple author details

### Recommendation Tools

- `get_paper_recommendations`: Paper recommendations with two modes:
  - Single-paper recommendations with pool selection
  - Multi-paper recommendations with positive/negative examples

## Usage Examples

### Basic Paper Search

```python
results = await paper_search(
    context,
    query="machine learning",
    year="2020-2024",
    min_citation_count=50,
    fields=["title", "abstract", "authors"]
)
```

### Advanced Search with Ranking

```python
results = await advanced_search_papers_semantic_scholar(
    context,
    query="transformer architecture",
    search_type="influential",
    filters={
        "year_range": (2020, 2024),
        "require_abstract": True,
        "citation_range": (100, None)
    }
)
```

### Paper Recommendations

```python
# Single paper recommendation
recommendations = await get_paper_recommendations(
    context,
    positive_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b"],
    fields="title,authors,year",
    limit=10,
    from_pool="recent"
)

# Multi-paper recommendation
recommendations = await get_paper_recommendations(
    context,
    positive_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"],
    negative_paper_ids=["ArXiv:1805.02262"],
    fields="title,abstract,authors"
)
```

### Batch Operations

```python
# Get details for multiple papers
papers = await paper_batch_details(
    context,
    paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"],
    fields="title,authors,year,citations"
)

# Get details for multiple authors
authors = await author_batch_details(
    context,
    author_ids=["1741101", "1780531"],
    fields="name,hIndex,citationCount,paperCount"
)
```

## Error Handling

The server provides standardized error responses:

```python
{
    "error": {
        "type": "error_type",  # rate_limit, api_error, validation, timeout
        "message": "Error description",
        "details": {
            # Additional context
            "authenticated": true/false  # Indicates if request was authenticated
        }
    }
}
```

## Best Practices

1. Use batch endpoints when requesting multiple items
2. Leverage predefined field combinations for common use cases
3. Use an API key for production environments
4. Implement appropriate rate limit handling
5. Validate inputs before making requests
6. Handle errors appropriately in your application

## Documentation

For detailed documentation of all tools and features, see [DOCUMENTATION.md](DOCUMENTATION.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
