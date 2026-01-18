"""Summarizer implementation for context management."""

import logging
from typing import Any, Dict, Optional

from spade_llm.providers.llm_provider import LLMProvider

logger = logging.getLogger("spade_llm.context.summarizer")


class Summarizer:
    """Wrapper class for LLM-based text summarization.
    
    This class provides a clean abstraction for summarizing text using an LLM provider,
    avoiding circular dependencies between context management and providers.
    """
    
    def __init__(self, provider: "LLMProvider"):
        """Initialize the summarizer with an LLM provider.
        
        Args:
            provider: An LLMProvider instance that will perform the summarization
        """
        self.provider = provider
        logger.info(f"Initialized Summarizer with provider: {provider.model}")
    
    async def summarize(
        self,
        text: str,
        prompt: str = "Summarize the following text concisely:",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a summary of the provided text using the LLM.
        
        Args:
            text: The text to summarize
            prompt: Instructions for the summarization (system prompt)
            metadata: Optional metadata for tracing (session_id, tags, etc.)
            
        Returns:
            The summarized text as a string
            
        Raises:
            Exception: If the summarization fails
        """
        try:
            logger.debug(f"Summarizing text (length: {len(text)} chars)")
            
            # Use the provider's summarize method
            summary = await self.provider.summarize(
                text=text,
                prompt=prompt,
                metadata=metadata
            )
            
            logger.info(f"Generated summary: {len(summary)} chars from {len(text)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            raise
