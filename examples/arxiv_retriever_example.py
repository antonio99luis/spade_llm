"""
Example usage of ArxivRetriever

This example demonstrates how to use the ArxivRetriever to search for
academic papers using the arXiv API.
"""

import asyncio
from spade_llm.rag.retrievers import ArxivRetriever


async def main():
    """Example of using ArxivRetriever."""
    
    # Initialize the retriever
    retriever = ArxivRetriever(
        max_results=10,
        sort_by="relevance",  # "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order="descending",  # "ascending" or "descending"
        timeout=30
    )
    
    # Basic search - search in all fields
    print("=== Basic Search: Neural Networks ===")
    query = "all:neural networks"
    documents = await retriever.retrieve(query, k=5)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Authors: {', '.join(doc.metadata.get('authors', []))}")
        print(f"arXiv ID: {doc.metadata.get('arxiv_id', 'N/A')}")
        print(f"Category: {doc.metadata.get('primary_category', 'N/A')}")
        print(f"Published: {doc.metadata.get('published', 'N/A')[:10]}")
        print(f"URL: {doc.metadata.get('url', 'N/A')}")
        print(f"Abstract: {doc.content[:200]}...")
    
    # Search by title
    print("\n\n=== Search by Title: Attention Mechanism ===")
    documents = await retriever.retrieve("ti:attention mechanism", k=3)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"arXiv ID: {doc.metadata.get('arxiv_id', 'N/A')}")
    
    # Search by author
    print("\n\n=== Search by Author: Goodfellow ===")
    documents = await retriever.retrieve("au:Goodfellow", k=3)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Authors: {', '.join(doc.metadata.get('authors', [])[:3])}...")
        print(f"Published: {doc.metadata.get('published', 'N/A')[:10]}")
    
    # Search by category (cs.AI - Artificial Intelligence)
    print("\n\n=== Search by Category: cs.AI with Transformers ===")
    documents = await retriever.retrieve("cat:cs.AI AND ti:transformers", k=3)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Category: {doc.metadata.get('primary_category', 'N/A')}")
        print(f"All Categories: {', '.join(doc.metadata.get('categories', []))}")
    
    # Retrieve specific papers by arXiv ID
    print("\n\n=== Retrieve by Specific arXiv IDs ===")
    # Famous papers: Attention Is All You Need, BERT, GPT-3
    arxiv_ids = ["1706.03762", "1810.04805", "2005.14165"]
    documents = await retriever.retrieve(
        query="",
        k=len(arxiv_ids),
        id_list=arxiv_ids
    )
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"arXiv ID: {doc.metadata.get('arxiv_id', 'N/A')}")
        print(f"Authors: {', '.join(doc.metadata.get('authors', [])[:5])}...")
        print(f"PDF: {doc.metadata.get('pdf_url', 'N/A')}")
    
    # Combined search with AND/OR operators
    print("\n\n=== Combined Search: Deep Learning AND Computer Vision ===")
    documents = await retriever.retrieve(
        "ti:deep learning AND cat:cs.CV",
        k=3
    )
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Primary Category: {doc.metadata.get('primary_category', 'N/A')}")
    
    # Recent papers (sort by submission date)
    print("\n\n=== Recent Papers in Machine Learning ===")
    recent_retriever = ArxivRetriever(
        max_results=5,
        sort_by="submittedDate",
        sort_order="descending"
    )
    documents = await recent_retriever.retrieve("cat:cs.LG", k=3)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Paper {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Submitted: {doc.metadata.get('published', 'N/A')[:10]}")
        print(f"Updated: {doc.metadata.get('updated', 'N/A')[:10]}")
    
    print("\n\n✅ All searches completed successfully!")
    print("\n📚 arXiv Query Syntax Tips:")
    print("  - ti:keyword     → Search in title")
    print("  - au:author      → Search by author")
    print("  - abs:keyword    → Search in abstract")
    print("  - cat:category   → Search by category")
    print("  - all:keyword    → Search in all fields")
    print("  - Use AND, OR, ANDNOT for complex queries")
    print("\n🔖 Popular arXiv Categories:")
    print("  - cs.AI  → Artificial Intelligence")
    print("  - cs.LG  → Machine Learning")
    print("  - cs.CV  → Computer Vision")
    print("  - cs.CL  → Computation and Language")
    print("  - cs.NE  → Neural and Evolutionary Computing")


if __name__ == "__main__":
    asyncio.run(main())
