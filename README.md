# Semantic Scholar MCP Server

A FastMCP server implementation for the Semantic Scholar API, providing comprehensive access to academic paper data, author information, and citation networks.

## Features

- **Paper Search & Discovery**

  - Full-text search with advanced filtering
  - Title-based paper matching
  - Paper recommendations with positive/negative examples
  - Batch paper details retrieval

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

## System Requirements

- Python 3.8+
- FastMCP framework
- `httpx` for async HTTP requests
- Environment variable for API key (optional)

## Installation

### Option 1: Claude Desktop

1. Clone this repository to your Claude Desktop workspace
2. Optionally set up the API key environment variable:
   ```bash
   export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
   ```
3. The server will be automatically available to Claude

### Option 2: Cline VSCode Plugin

1. Copy the server implementation to your workspace
2. Optionally configure the API key in your environment
3. The server will be accessible through the Cline plugin

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

- `paper_search`: General paper search with filters
- `paper_search_match`: Exact title matching
- `paper_details`: Single paper details
- `paper_batch_details`: Multiple paper details

### Citation Tools

- `paper_citations`: Get citing papers
- `paper_references`: Get referenced papers

### Author Tools

- `author_search`: Search for authors
- `author_details`: Get author information
- `author_papers`: Get author's publications
- `author_batch_details`: Multiple author details

### Advanced Tools

- `advanced_search_papers_semantic_scholar`: Complex search with ranking
- `get_paper_recommendations`: Paper recommendations

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

### Get Paper Recommendations

```python
recommendations = await get_paper_recommendations(
    context,
    positive_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b"],
    fields="title,authors,year",
    limit=10
)
```

### Batch Author Details

```python
author_info = await author_batch_details(
    context,
    author_ids=["1741101", "1780531"],
    fields="name,hIndex,citationCount,paperCount"
)
```

## Documentation

For detailed documentation of all tools and features, see [DOCUMENTATION.md](DOCUMENTATION.md).

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
