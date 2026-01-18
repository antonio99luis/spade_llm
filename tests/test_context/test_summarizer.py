"""Tests for Summarizer class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from spade_llm.context.summarizer import Summarizer


class TestSummarizerInitialization:
    """Test Summarizer initialization."""
    
    def test_init_with_provider(self):
        """Test initialization with an LLM provider."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        
        summarizer = Summarizer(provider=mock_provider)
        
        assert summarizer.provider == mock_provider
    
    def test_init_stores_provider_reference(self):
        """Test that provider reference is stored correctly."""
        mock_provider = Mock()
        mock_provider.model = "claude-3-opus"
        
        summarizer = Summarizer(provider=mock_provider)
        
        assert hasattr(summarizer, 'provider')
        assert summarizer.provider is mock_provider


class TestSummarizerSummarize:
    """Test Summarizer.summarize() method."""
    
    @pytest.mark.asyncio
    async def test_summarize_basic(self):
        """Test basic summarization."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="This is a summary")
        
        summarizer = Summarizer(provider=mock_provider)
        
        text = "This is a long text that needs to be summarized."
        result = await summarizer.summarize(text=text)
        
        assert result == "This is a summary"
        mock_provider.summarize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summarize_with_custom_prompt(self):
        """Test summarization with custom prompt."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="Custom summary")
        
        summarizer = Summarizer(provider=mock_provider)
        
        text = "Text to summarize"
        custom_prompt = "Summarize this in one sentence:"
        result = await summarizer.summarize(text=text, prompt=custom_prompt)
        
        assert result == "Custom summary"
        # Verify the custom prompt was passed to provider
        call_args = mock_provider.summarize.call_args
        assert call_args.kwargs['prompt'] == custom_prompt
    
    @pytest.mark.asyncio
    async def test_summarize_with_metadata(self):
        """Test summarization with metadata for tracing."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="Summary with metadata")
        
        summarizer = Summarizer(provider=mock_provider)
        
        text = "Text to summarize"
        metadata = {"session_id": "test-123", "tags": ["test"]}
        result = await summarizer.summarize(text=text, metadata=metadata)
        
        assert result == "Summary with metadata"
        # Verify metadata was passed
        call_args = mock_provider.summarize.call_args
        assert call_args.kwargs['metadata'] == metadata
    
    @pytest.mark.asyncio
    async def test_summarize_default_prompt(self):
        """Test summarization uses default prompt when none provided."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="Summary")
        
        summarizer = Summarizer(provider=mock_provider)
        
        text = "Text to summarize"
        await summarizer.summarize(text=text)
        
        # Verify default prompt was used
        call_args = mock_provider.summarize.call_args
        assert 'prompt' in call_args.kwargs
        assert "concisely" in call_args.kwargs['prompt']
    
    @pytest.mark.asyncio
    async def test_summarize_passes_all_args_to_provider(self):
        """Test that all arguments are passed correctly to provider."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="Result")
        
        summarizer = Summarizer(provider=mock_provider)
        
        text = "Test text"
        prompt = "Custom prompt"
        metadata = {"key": "value"}
        
        await summarizer.summarize(text=text, prompt=prompt, metadata=metadata)
        
        mock_provider.summarize.assert_called_once_with(
            text=text,
            prompt=prompt,
            metadata=metadata
        )
    
    @pytest.mark.asyncio
    async def test_summarize_handles_long_text(self):
        """Test summarization with long text input."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="Short summary")
        
        summarizer = Summarizer(provider=mock_provider)
        
        # Create a long text (10,000 chars)
        long_text = "Lorem ipsum dolor sit amet. " * 400
        result = await summarizer.summarize(text=long_text)
        
        assert result == "Short summary"
        # Verify long text was passed
        call_args = mock_provider.summarize.call_args
        assert len(call_args.kwargs['text']) > 5000
    
    @pytest.mark.asyncio
    async def test_summarize_error_handling(self):
        """Test error handling when summarization fails."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(side_effect=Exception("API Error"))
        
        summarizer = Summarizer(provider=mock_provider)
        
        with pytest.raises(Exception) as exc_info:
            await summarizer.summarize(text="Test text")
        
        assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_summarize_empty_text(self):
        """Test summarization with empty text."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        mock_provider.summarize = AsyncMock(return_value="")
        
        summarizer = Summarizer(provider=mock_provider)
        
        result = await summarizer.summarize(text="")
        
        # Should still call provider even with empty text
        mock_provider.summarize.assert_called_once()
        assert result == ""


class TestSummarizerProtocolCompliance:
    """Test that Summarizer implements SummarizerProtocol correctly."""
    
    def test_has_summarize_method(self):
        """Test that Summarizer has the summarize method."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        
        summarizer = Summarizer(provider=mock_provider)
        
        assert hasattr(summarizer, 'summarize')
        assert callable(summarizer.summarize)
    
    @pytest.mark.asyncio
    async def test_summarize_is_async(self):
        """Test that summarize method is async."""
        import inspect
        
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        
        summarizer = Summarizer(provider=mock_provider)
        
        assert inspect.iscoroutinefunction(summarizer.summarize)
    
    @pytest.mark.asyncio
    async def test_protocol_signature_matches(self):
        """Test that summarize signature matches SummarizerProtocol."""
        import inspect
        from spade_llm.context._types import SummarizerProtocol
        
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        
        summarizer = Summarizer(provider=mock_provider)
        
        # Get signature of the method
        sig = inspect.signature(summarizer.summarize)
        
        # Verify expected parameters exist
        assert 'text' in sig.parameters
        assert 'prompt' in sig.parameters
        assert 'metadata' in sig.parameters


class TestSummarizerIntegration:
    """Integration tests for Summarizer with mock providers."""
    
    @pytest.mark.asyncio
    async def test_summarize_conversation_format(self):
        """Test summarizing a conversation format text."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        expected_summary = "User asked about weather, assistant provided forecast."
        mock_provider.summarize = AsyncMock(return_value=expected_summary)
        
        summarizer = Summarizer(provider=mock_provider)
        
        conversation_text = """
        USER: What's the weather like today?
        ASSISTANT: It's sunny with a temperature of 25°C.
        USER: Will it rain tomorrow?
        ASSISTANT: Yes, there's a 70% chance of rain tomorrow afternoon.
        """
        
        result = await summarizer.summarize(text=conversation_text)
        
        # Verify the summarizer returned what the provider returned
        assert result == expected_summary
        # Verify the provider was called with the conversation text
        call_args = mock_provider.summarize.call_args
        assert call_args.kwargs['text'] == conversation_text
        # Verify the result is a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_summarize_with_system_context(self):
        """Test summarizing with system context included."""
        mock_provider = Mock()
        mock_provider.model = "gpt-4"
        expected_summary = "Summary with context"
        mock_provider.summarize = AsyncMock(return_value=expected_summary)
        
        summarizer = Summarizer(provider=mock_provider)
        
        text_with_context = "SYSTEM: You are a helpful assistant.\nUSER: Hello!"
        prompt = "Summarize keeping system context in mind"
        
        result = await summarizer.summarize(text=text_with_context, prompt=prompt)
        
        # Verify the result is what the provider returned
        assert result == expected_summary
        # Verify the provider was called with correct arguments
        call_args = mock_provider.summarize.call_args
        assert call_args.kwargs['text'] == text_with_context
        assert call_args.kwargs['prompt'] == prompt
        # Verify the result is a valid string
        assert isinstance(result, str)
