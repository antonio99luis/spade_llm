"""
Example usage of TavilyRetriever

This example demonstrates how to use the TavilyRetriever to search for
web content using the Tavily Search API.
"""

import asyncio
import os
from spade_llm.rag.retrievers import TavilyRetriever
from spade_llm.utils.env_loader import load_env_vars

# Load environment variables from .env file
load_env_vars()


async def main():
    """Example of using TavilyRetriever."""
    
    # Initialize the retriever
    # Make sure to set your TAVILY_API_KEY environment variable
    retriever = TavilyRetriever(
        #api_key="tvly-XXXXXXXXXXXXX",  # Optional if using env var
        search_depth="basic",  # or "advanced"
        include_images=False,
        include_answer=True,
        max_results=10,
        topic="general",  # Topic for the search
        days=2,  # Number of days to search back
        timeout=100  # Request timeout in seconds
    )
    
    # Basic search
    print("=== Basic Search ===")
    query = "latest developments in artificial intelligence 2024"
    documents = await retriever.retrieve(query, k=5)
    
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"URL: {doc.metadata.get('url', 'N/A')}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Content: {doc.content[:200]}...")
    
    # Search with domain filtering
    print("\n\n=== Search with Domain Filtering ===")
    documents = await retriever.retrieve(
        query="machine learning tutorials",
        k=3,
        include_domains=["github.com", "arxiv.org"],
        exclude_domains=["example.com"]
    )
    
    for i, doc in enumerate(documents, 1):
        print(f"\nFiltered Document {i}:")
        print(f"URL: {doc.metadata.get('url', 'N/A')}")
        print(f"Content: {doc.content[:150]}...")
    
    # Advanced search
    print("\n\n=== Advanced Search ===")
    documents = await retriever.search_advanced(
        query="quantum computing breakthroughs",
        k=3
    )
    
    for i, doc in enumerate(documents, 1):
        print(f"\nAdvanced Document {i}:")
        print(f"URL: {doc.metadata.get('url', 'N/A')}")
        print(f"Content: {doc.content[:150]}...")


if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("TAVILY_API_KEY"):
        print("Please set your TAVILY_API_KEY environment variable")
        print("You can get an API key from https://tavily.com")
    else:
        asyncio.run(main())