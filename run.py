#!/usr/bin/env python3
"""
Entry point script for the Semantic Scholar API Server.

Available tools:
- paper_relevance_search
- paper_bulk_search
- paper_title_search
- paper_details
- paper_batch_details
- paper_authors
- paper_citations
- paper_references
- author_search
- author_details
- author_papers
- author_batch_details
- get_paper_recommendations_single
- get_paper_recommendations_multi
"""

# Import the mcp instance from centralized location
from semantic_scholar.mcp import mcp
# Import the main function from server
from semantic_scholar.server import main

# Import all API modules to ensure tools are registered
from semantic_scholar.api import papers, authors, recommendations

if __name__ == "__main__":
    main() 