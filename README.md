# Semantic Scholar MCP Server

An MCP server that provides access to the Semantic Scholar API through Model Context Protocol (MCP). This server is built with the FastMCP framework, enabling LLMs to search and retrieve academic paper information, author details, and perform advanced scholarly literature analysis.

## 📋 System Requirements

- Python 3.8+
- pip (Python package manager)
- Semantic Scholar API key

## 📦 Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Required Packages

- **fastmcp**: Framework for building Model Context Protocol servers
- **httpx**: Async HTTP client for API requests
- **python-dotenv**: Environment variable management

All dependencies are specified in `requirements.txt` for easy installation.

## 📑 Table of Contents

- [System Requirements](#-system-requirements)
- [Dependencies](#-dependencies)
- [MCP Tools](#%EF%B8%8F-mcp-tools)
- [Getting Started](#-getting-started)
- [Installation Options](#-installation-options)
  - [Claude Desktop](#option-1-install-for-claude-desktop)
  - [Cline VSCode Plugin](#option-2-install-for-cline-vscode-plugin)
- [Environment Variables](#%EF%B8%8F-environment-variables)

## 🛠️ MCP Tools

The server exposes the following tools to LLMs:

### custom_search_papers_semantic_scholar

Search for academic papers with customizable fields and limits:

- Query-based paper search
- Configurable result fields
- Pagination support
- Citation information

### get_paper_details_semantic_scholar

Retrieve detailed information about specific papers:

- Full paper metadata
- Citation counts
- Author information
- References and citations
- Venue details

### get_author_details_semantic_scholar

Access comprehensive author information:

- Publication metrics
- H-index
- Citation counts
- Publication history

### advanced_search_papers_semantic_scholar

Advanced search functionality with comprehensive filtering options:

- Year range filtering
- Citation threshold filtering
- Multiple sorting options (relevance, citations, year, influence)
- Customizable result limits
- Combined functionality for finding both recent and seminal papers

## 🚀 Getting Started

Clone the repository:

```bash
git clone [repository-url]
cd semantic-scholar-server
```

## 📦 Installation Options

You can install this MCP server in either Claude Desktop or the Cline VSCode plugin.

### Option 1: Install for Claude Desktop

Install using FastMCP:

```bash
fastmcp install semantic-scholar-server.py --name "Semantic Scholar Server" -e SEMANTIC_SCHOLAR_API_KEY=your_api_key
```

Replace `your_api_key` with your Semantic Scholar API key.

### Option 2: Install for Cline VSCode Plugin

To use this server with the [Cline VSCode plugin](http://cline.bot):

1. In VSCode, click the server icon (☰) in the Cline plugin sidebar
2. Click the "Edit MCP Settings" button (✎)
3. Add the following configuration to the settings file:

```json
{
  "semantic-scholar": {
    "command": "uv",
    "args": [
      "run",
      "--with",
      "fastmcp",
      "fastmcp",
      "run",
      "/path/to/repo/semantic-scholar-server.py"
    ],
    "env": {
      "SEMANTIC_SCHOLAR_API_KEY": "your_api_key"
    }
  }
}
```

Replace:

- `/path/to/repo` with the full path to where you cloned this repository
- `your_api_key` with your Semantic Scholar API key

## ⚙️ Environment Variables

The following environment variables must be set:

- `SEMANTIC_SCHOLAR_API_KEY`: Your Semantic Scholar API key (required for accessing the API)
