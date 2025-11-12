"""arXiv search API retriever implementation."""

import logging
import re
from typing import List, Optional, Dict, Any, Literal
import asyncio
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

from .base import BaseRetriever
from ..core.document import Document

logger = logging.getLogger(__name__)


class ArxivRetriever(BaseRetriever):
    """Retriever implementation using arXiv API for academic paper search.
    
    This retriever uses the arXiv API to find relevant academic papers
    based on search queries. It's particularly useful for retrieving
    research papers, preprints, and academic content from arXiv.
    
    The retriever returns paper information as Document objects with metadata
    including paper IDs, authors, categories, and publication dates.
    """
    
    def __init__(
        self,
        max_results: int = 10,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
        sort_order: Literal["ascending", "descending"] = "descending",
        timeout: int = 30
    ):
        """Initialize the arXiv retriever.
        
        Args:
            max_results: Maximum number of results to return from API (default 10)
            sort_by: Sort criteria - "relevance", "lastUpdatedDate", or "submittedDate"
            sort_order: Sort order - "ascending" or "descending"
            timeout: Request timeout in seconds (default 30)
        """
        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.timeout = timeout
        self.base_url = "http://export.arxiv.org/api/query"
        
        logger.info(
            f"Initialized ArxivRetriever with max_results={max_results}, "
            f"sort_by={sort_by}, sort_order={sort_order}"
        )
    
    def _build_query_url(
        self,
        query: str,
        max_results: int,
        id_list: Optional[List[str]] = None,
        start: int = 0
    ) -> str:
        """Build the arXiv API query URL.
        
        Args:
            query: The search query
            max_results: Maximum number of results
            id_list: Optional list of arXiv IDs to retrieve
            start: Start index for pagination
            
        Returns:
            Complete API URL string
        """
        params = {
            "max_results": max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order,
            "start": start
        }
        
        if id_list:
            # Search by specific IDs
            params["id_list"] = ",".join(id_list)
        else:
            # Search by query
            params["search_query"] = query
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.base_url}?{query_string}"
    
    def _search(
        self,
        query: str,
        max_results: int,
        id_list: Optional[List[str]] = None,
        start: int = 0
    ) -> ET.Element:
        """Internal search method to query the arXiv API.
        
        Args:
            query: The search query
            max_results: Maximum number of results
            id_list: Optional list of arXiv IDs
            start: Start index for pagination
            
        Returns:
            XML Element tree root from API response
            
        Raises:
            Exception: If the API request fails
        """
        url = self._build_query_url(query, max_results, id_list, start)
        
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                return root
        except Exception as e:
            logger.error(f"Error querying arXiv API: {e}")
            raise
    
    def _parse_entry(self, entry: ET.Element, query: str) -> Optional[Document]:
        """Parse an arXiv entry XML element into a Document.
        
        Args:
            entry: XML element representing a paper entry
            query: The original search query
            
        Returns:
            Document object or None if parsing fails
        """
        # Namespace for arXiv API
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        try:
            # Extract basic information
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            title = re.sub(r'\s+', ' ', title)  # Clean up whitespace
            
            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            summary = re.sub(r'\s+', ' ', summary)  # Clean up whitespace
            
            # Extract ID and URL
            id_elem = entry.find('atom:id', ns)
            paper_id = id_elem.text if id_elem is not None else ""
            arxiv_id = paper_id.split('/abs/')[-1] if paper_id else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract dates
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text if published_elem is not None else ""
            
            updated_elem = entry.find('atom:updated', ns)
            updated = updated_elem.text if updated_elem is not None else ""
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Primary category
            primary_category_elem = entry.find('arxiv:primary_category', ns)
            primary_category = (primary_category_elem.get('term') 
                              if primary_category_elem is not None else "")
            
            # PDF link
            pdf_link = ""
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href', '')
                    break
            
            # Build document content
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if authors:
                content_parts.append(f"Authors: {', '.join(authors)}")
            if summary:
                content_parts.append(f"\nAbstract:\n{summary}")
            
            content = "\n".join(content_parts)
            
            if not content.strip():
                return None
            
            # Build metadata
            metadata = {
                "source": "arxiv",
                "query": query,
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "updated": updated,
                "categories": categories,
                "primary_category": primary_category,
                "url": paper_id,
                "pdf_url": pdf_link,
            }
            
            # Create document
            return Document(content=content, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        id_list: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve papers using arXiv API.
        
        Args:
            query: The search query (supports arXiv query syntax)
            k: Number of papers to retrieve (will be min of k and max_results)
            id_list: Optional list of specific arXiv IDs to retrieve
            **kwargs: Additional parameters:
                - start: Start index for pagination (default 0)
                - categories: Filter by categories (not implemented in base query)
            
        Returns:
            List of Document objects with paper information
            
        Example queries:
            - "ti:machine learning" - Search in title
            - "au:Goodfellow" - Search by author
            - "cat:cs.AI" - Search by category
            - "all:neural networks" - Search in all fields
            - "ti:quantum AND cat:quant-ph" - Combined search
        """
        try:
            logger.debug(f"Searching arXiv for query: {query[:100]}...")
            
            # Limit k to max_results
            k = min(k, self.max_results)
            start = kwargs.get("start", 0)
            
            # Use asyncio to run the synchronous request in a thread pool
            loop = asyncio.get_event_loop()
            root = await loop.run_in_executor(
                None,
                self._search,
                query,
                k,
                id_list,
                start
            )
            
            # Parse XML response
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            # Convert entries to Documents
            documents = []
            for entry in entries:
                doc = self._parse_entry(entry, query)
                if doc:
                    documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} papers from arXiv")
            return documents
            
        except Exception as e:
            logger.error(f"Error during arXiv retrieval: {e}")
            raise
    

