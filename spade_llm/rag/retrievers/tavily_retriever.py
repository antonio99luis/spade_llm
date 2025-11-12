"""Tavily search API retriever implementation."""

import logging
import os
from typing import List, Optional, Dict, Any, Literal, Sequence
import asyncio
import requests
import json

from .base import BaseRetriever
from ..core.document import Document

logger = logging.getLogger(__name__)


class TavilyRetriever(BaseRetriever):
    """Retriever implementation using Tavily Search API for web search.
    
    This retriever uses the Tavily Search API to find relevant web content
    based on search queries. It's particularly useful for retrieving up-to-date
    information from the web that may not be available in local document stores.
    
    The retriever returns web search results as Document objects with metadata
    including URLs, snippets, and search relevance scores.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_images: bool = False,
        include_answer: bool = True,
        max_results: int = 10,
        topic: str = "general",
        days: int = 2,
        timeout: int = 100
    ):
        """Initialize the Tavily retriever.
        
        Args:
            api_key: Tavily API key. If None, will try to get from TAVILY_API_KEY env var
            search_depth: Search depth ("basic" or "advanced")
            include_images: Whether to include image results
            include_answer: Whether to include AI-generated answer
            max_results: Maximum number of results to return from API
            topic: Topic for the search (default "general")
            days: Number of days to search back (default 2)
            timeout: Request timeout in seconds (default 100)
        """
        self.api_key = self._get_api_key(api_key)
        self.search_depth = search_depth
        self.include_images = include_images
        self.include_answer = include_answer
        self.max_results = max_results
        self.topic = topic
        self.days = days
        self.timeout = timeout
        self.base_url = "https://api.tavily.com/search"
        self.headers = {
            "Content-Type": "application/json",
        }
        
        logger.info(
            f"Initialized TavilyRetriever with search_depth={search_depth}, "
            f"max_results={max_results}, topic={topic}"
        )
    
    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the Tavily API key from parameter or environment variable.
        
        Args:
            api_key: Optional API key parameter
            
        Returns:
            The API key string
            
        Raises:
            ValueError: If no API key is found
        """
        if api_key:
            return api_key
            
        try:
            return os.environ["TAVILY_API_KEY"]
        except KeyError:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def _search(
        self,
        query: str,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        topic: Optional[str] = None,
        days: Optional[int] = None,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        include_answer: Optional[bool] = None,
        include_raw_content: bool = False,
        include_images: Optional[bool] = None,
        use_cache: bool = True,
        **kwargs
    ) -> dict:
        """Internal search method to send the request to the API.
        
        Args:
            query: The search query
            search_depth: Search depth ("basic" or "advanced")
            topic: Topic for the search
            days: Number of days to search back
            max_results: Maximum number of results
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            include_answer: Whether to include AI answer
            include_raw_content: Whether to include raw content
            include_images: Whether to include images
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            API response as dictionary
        """
        # Use instance defaults if not provided
        data = {
            "query": query,
            "search_depth": search_depth or self.search_depth,
            "topic": topic or self.topic,
            "days": days or self.days,
            "include_answer": include_answer if include_answer is not None else self.include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results or self.max_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_images": include_images if include_images is not None else self.include_images,
            "api_key": self.api_key,
            "use_cache": use_cache,
        }
        
        # Add any additional kwargs
        data.update(kwargs)

        response = requests.post(
            self.base_url, 
            data=json.dumps(data), 
            headers=self.headers, 
            timeout=self.timeout
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using Tavily search API.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (will be min of k and max_results)
            search_depth: Override default search depth ("basic" or "advanced")
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            **kwargs: Additional parameters for the API request
            
        Returns:
            List of Document objects with web search results
        """
        try:
            logger.debug(f"Searching Tavily for query: {query[:100]}...")
            
            # Limit k to max_results
            k = min(k, self.max_results)
            
            # Use asyncio to run the synchronous request in a thread pool
            # to maintain async compatibility
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._search,
                query,
                search_depth,
                kwargs.get("topic"),
                kwargs.get("days"),
                k,
                include_domains,
                exclude_domains,
                kwargs.get("include_answer"),
                kwargs.get("include_raw_content", False),
                kwargs.get("include_images"),
                kwargs.get("use_cache", True)
            )
            
            # Convert results to Document objects
            documents = self._convert_results_to_documents(data, query)
            
            logger.debug(f"Retrieved {len(documents)} documents from Tavily")
            return documents
            
        except Exception as e:
            logger.error(f"Error during Tavily retrieval: {e}")
            raise
    
    def _convert_results_to_documents(
        self, 
        api_response: Dict[str, Any], 
        original_query: str
    ) -> List[Document]:
        """Convert Tavily API response to Document objects.
        
        Args:
            api_response: Raw response from Tavily API
            original_query: The original search query
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Process search results
        results = api_response.get("results", [])
        
        for i, result in enumerate(results):
            # Extract content and metadata
            content = result.get("content", "")
            title = result.get("title", "")
            url = result.get("url", "")
            score = result.get("score", 0.0)
            
            # Combine title and content for document content
            if title and content:
                document_content = f"{title}\n\n{content}"
            elif title:
                document_content = title
            else:
                document_content = content
            
            # Skip empty results
            if not document_content.strip():
                continue
            
            # Create metadata
            metadata = {
                "source": "tavily_search",
                "query": original_query,
                "url": url,
                "title": title,
                "score": score,
                "result_index": i,
                "search_timestamp": api_response.get("search_time", ""),
            }
            
            # Add any additional fields from the result
            for key, value in result.items():
                if key not in ["content", "title", "url", "score"]:
                    metadata[f"tavily_{key}"] = value
            
            # Create document
            document = Document(
                content=document_content,
                metadata=metadata
            )
            documents.append(document)
        
        # If there's an AI-generated answer, add it as the first document
        if self.include_answer and "answer" in api_response:
            answer = api_response["answer"]
            if answer and answer.strip():
                answer_doc = Document(
                    content=answer,
                    metadata={
                        "source": "tavily_answer",
                        "query": original_query,
                        "type": "ai_generated_answer",
                        "search_timestamp": api_response.get("search_time", ""),
                    }
                )
                # Insert answer at the beginning
                documents.insert(0, answer_doc)
        
        return documents