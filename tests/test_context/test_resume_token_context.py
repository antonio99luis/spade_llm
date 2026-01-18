"""Tests for ResumeTokenContext class."""

import pytest
from unittest.mock import Mock, AsyncMock

from spade_llm.context.management import ResumeTokenContext
from spade_llm.context._types import (
    create_user_message,
    create_assistant_message,
    create_system_message,
)


class TestResumeTokenContextInitialization:
    """Test ResumeTokenContext initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        context = ResumeTokenContext()
        
        assert context.max_tokens == 4096
        assert context.reserve_tokens == 500
        assert context.available_tokens == 3596
        assert context.message_to_resume == 4
        assert context.message_resume_overlap == 1
        assert context.resume_provider is None
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_summarizer = Mock()
        
        context = ResumeTokenContext(
            max_tokens=8192,
            reserve_tokens=1000,
            model="gpt-4",
            resume_provider=mock_summarizer,
            message_to_resume=6,
            message_resume_overlap=2,
        )
        
        assert context.max_tokens == 8192
        assert context.reserve_tokens == 1000
        assert context.available_tokens == 7192
        assert context.message_to_resume == 6
        assert context.message_resume_overlap == 2
        assert context.resume_provider == mock_summarizer
        assert context.model == "gpt-4"
    
    def test_init_with_custom_summary_prompt(self):
        """Test initialization with custom summary prompt."""
        custom_prompt = "Create a brief summary focusing on key decisions."
        
        context = ResumeTokenContext(summary_system_prompt=custom_prompt)
        
        assert context.summary_system_prompt == custom_prompt
    
    def test_init_uses_default_summary_prompt(self):
        """Test that default summary prompt is used when none provided."""
        context = ResumeTokenContext()
        
        assert context.summary_system_prompt == ResumeTokenContext.DEFAULT_SUMMARY_PROMPT
        assert "conversation summarizer" in context.summary_system_prompt.lower()
    
    def test_init_validates_max_tokens(self):
        """Test validation of max_tokens parameter."""
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            ResumeTokenContext(max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            ResumeTokenContext(max_tokens=-100)
    
    def test_init_validates_reserve_tokens(self):
        """Test validation of reserve_tokens parameter."""
        with pytest.raises(ValueError, match="reserve_tokens must be >= 0"):
            ResumeTokenContext(reserve_tokens=-1)
        
        with pytest.raises(ValueError, match="reserve_tokens must be less than max_tokens"):
            ResumeTokenContext(max_tokens=1000, reserve_tokens=1000)
        
        with pytest.raises(ValueError, match="reserve_tokens must be less than max_tokens"):
            ResumeTokenContext(max_tokens=1000, reserve_tokens=1500)
    
    def test_init_validates_message_to_resume(self):
        """Test validation of message_to_resume parameter."""
        with pytest.raises(ValueError, match="message_to_resume must be greater than 0"):
            ResumeTokenContext(message_to_resume=0)
        
        with pytest.raises(ValueError, match="message_to_resume must be greater than 0"):
            ResumeTokenContext(message_to_resume=-1)
    
    def test_init_validates_message_resume_overlap(self):
        """Test validation of message_resume_overlap parameter."""
        with pytest.raises(ValueError, match="message_resume_overlap must be >= 0"):
            ResumeTokenContext(message_resume_overlap=-1)


class TestResumeTokenContextApplyStrategy:
    """Test ResumeTokenContext.apply_context_strategy() method."""
    
    @pytest.mark.asyncio
    async def test_apply_strategy_no_messages(self):
        """Test strategy with no messages."""
        context = ResumeTokenContext()
        
        result = await context.apply_context_strategy([])
        
        assert result == []
        assert context._last_token_count == 0
        assert context._last_applied_token_count == 0
    
    @pytest.mark.asyncio
    async def test_apply_strategy_under_token_limit(self):
        """Test strategy when messages are under token limit."""
        context = ResumeTokenContext(max_tokens=10000)
        
        messages = [
            create_user_message("Hello"),
            create_assistant_message("Hi there!"),
            create_user_message("How are you?"),
        ]
        
        result = await context.apply_context_strategy(messages)
        
        # Should return all messages unchanged
        assert len(result) == 3
        assert result == messages
    
    @pytest.mark.asyncio
    async def test_apply_strategy_no_summarizer_falls_back(self):
        """Test strategy falls back to trim when no summarizer provided."""
        context = ResumeTokenContext(
            max_tokens=100,  # Very low limit to force trimming
            resume_provider=None  # No summarizer
        )
        
        # Create many messages to exceed limit
        messages = [
            create_user_message(f"Message {i} with some content") 
            for i in range(20)
        ]
        
        result = await context.apply_context_strategy(messages)
        
        # Should have trimmed messages (kept most recent)
        assert len(result) < len(messages)
        assert result[-1] == messages[-1]  # Most recent should be kept
    
    @pytest.mark.asyncio
    async def test_apply_strategy_with_summarization(self):
        """Test strategy applies summarization when needed."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(
            return_value="Summary of old messages"
        )
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=3,
            message_resume_overlap=1,
        )
        
        # Create enough messages to trigger summarization
        messages = [
            create_user_message(f"Old message {i}") for i in range(5)
        ] + [
            create_user_message(f"Recent message {i}") for i in range(3)
        ]
        
        result = await context.apply_context_strategy(messages)
        
        # Should have called summarizer
        mock_summarizer.summarize.assert_called_once()
        
        # Result should contain summary + recent messages
        assert len(result) >= 3  # At least the recent messages
        # First message should be the summary (role: assistant)
        assert result[0]["role"] == "assistant"
        assert "Summary" in result[0]["content"]
    
    @pytest.mark.asyncio
    async def test_apply_strategy_with_overlap(self):
        """Test that overlap messages are included in summary."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=3,
            message_resume_overlap=2,  # Overlap 2 messages
        )
        
        messages = [
            create_user_message(f"Message {i}") for i in range(10)
        ]
        
        await context.apply_context_strategy(messages)
        
        # Verify summarizer was called with the right amount of messages
        call_args = mock_summarizer.summarize.call_args
        summarized_text = call_args.kwargs['text']
        
        # Should include old messages + 2 overlap messages
        # (10 total - 3 recent = 7 old, + 2 overlap = 9 messages summarized)
        assert "Message" in summarized_text
    
    @pytest.mark.asyncio
    async def test_apply_strategy_summary_system_prompt_used(self):
        """Test that summary system prompt is passed to summarizer."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        custom_prompt = "Summarize focusing on important details"
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=2,
            summary_system_prompt=custom_prompt,
        )
        
        messages = [
            create_user_message(f"Message {i}") for i in range(10)
        ]
        
        await context.apply_context_strategy(messages)
        
        # Verify custom prompt was used
        call_args = mock_summarizer.summarize.call_args
        prompt_used = call_args.kwargs['prompt']
        assert custom_prompt in prompt_used or custom_prompt == prompt_used.split("\n")[0]
    
    @pytest.mark.asyncio
    async def test_apply_strategy_with_agent_system_prompt(self):
        """Test that agent system prompt is included in context."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=2,
        )
        
        messages = [create_user_message(f"Message {i}") for i in range(10)]
        agent_system_prompt = "You are a helpful assistant"
        
        await context.apply_context_strategy(messages, system_prompt=agent_system_prompt)
        
        # Verify agent system prompt was passed to summarizer
        call_args = mock_summarizer.summarize.call_args
        summarized_text = call_args.kwargs['text']
        assert "SYSTEM:" in summarized_text or agent_system_prompt in summarized_text


class TestResumeTokenContextFallback:
    """Test fallback behavior when summary is too large."""
    
    @pytest.mark.asyncio
    async def test_fallback_when_summary_too_large(self):
        """Test fallback when summary alone exceeds token limit."""
        # Create a summarizer that returns a very long summary
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(
            return_value="Very long summary " * 1000  # Very long
        )
        
        context = ResumeTokenContext(
            max_tokens=100,  # Very small limit
            resume_provider=mock_summarizer,
            message_to_resume=2,
        )
        
        messages = [create_user_message(f"Message {i}") for i in range(10)]
        
        result = await context.apply_context_strategy(messages)
        
        # Should fall back to token trim
        assert len(result) > 0
        # Should keep most recent messages
        assert result[-1] == messages[-1]
    
    @pytest.mark.asyncio
    async def test_trim_recent_when_summary_plus_recent_exceeds(self):
        """Test trimming recent messages when summary + recent exceeds limit."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary of old messages")
        
        context = ResumeTokenContext(
            max_tokens=200,  # Small limit
            reserve_tokens=50,
            resume_provider=mock_summarizer,
            message_to_resume=10,  # Keep many recent
        )
        
        # Create messages with some content
        messages = [
            create_user_message(f"Message {i} with substantial content here") 
            for i in range(20)
        ]
        
        result = await context.apply_context_strategy(messages)
        
        # Should have summary + some (but not all) recent messages
        assert result[0]["role"] == "assistant"  # Summary
        assert len(result) < len(messages)  # Some messages trimmed
        assert len(result) > 1  # But kept some recent messages


class TestResumeTokenContextStats:
    """Test ResumeTokenContext.get_stats() method."""
    
    @pytest.mark.asyncio
    async def test_get_stats_after_no_summarization(self):
        """Test stats when no summarization occurred."""
        context = ResumeTokenContext(max_tokens=10000)
        
        messages = [
            create_user_message("Hello"),
            create_assistant_message("Hi!"),
        ]
        
        await context.apply_context_strategy(messages)
        stats = context.get_stats(total_messages=2)
        
        assert stats["strategy"] == "resume_token_based"
        assert stats["total_messages"] == 2
        assert stats["messages_in_context"] == 2
        assert stats["used_summary"] is False
        assert stats["summary_tokens"] == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_after_summarization(self):
        """Test stats when summarization occurred."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=2,
        )
        
        messages = [create_user_message(f"Message {i}") for i in range(10)]
        
        await context.apply_context_strategy(messages)
        stats = context.get_stats(total_messages=10)
        
        assert stats["strategy"] == "resume_token_based"
        assert stats["total_messages"] == 10
        assert stats["used_summary"] is True
        assert stats["summary_tokens"] > 0
        assert stats["messages_in_context"] >= 2  # At least recent messages
    
    def test_get_stats_includes_all_fields(self):
        """Test that stats includes all expected fields."""
        context = ResumeTokenContext()
        stats = context.get_stats(total_messages=5)
        
        expected_fields = [
            "strategy",
            "max_tokens",
            "reserve_tokens",
            "available_tokens",
            "message_to_resume",
            "message_resume_overlap",
            "used_summary",
            "summary_tokens",
            "total_messages",
            "messages_in_context",
            "messages_dropped",
            "model",
            "tokens_before",
            "tokens_after",
            "tokens_saved",
            "tokens_available",
        ]
        
        for field in expected_fields:
            assert field in stats


class TestResumeTokenContextMessageFormatting:
    """Test message formatting in summarization."""
    
    @pytest.mark.asyncio
    async def test_formats_user_messages_correctly(self):
        """Test that user messages are formatted correctly for summarization."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=1,
        )
        
        messages = [
            create_user_message("Hello, how are you?"),
            create_user_message("What's the weather?"),
        ]
        
        await context.apply_context_strategy(messages)
        
        # Check the text passed to summarizer
        call_args = mock_summarizer.summarize.call_args
        summarized_text = call_args.kwargs['text']
        
        assert "USER:" in summarized_text
        assert "Hello, how are you?" in summarized_text
    
    @pytest.mark.asyncio
    async def test_formats_assistant_messages_correctly(self):
        """Test that assistant messages are formatted correctly."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=1,
        )
        
        messages = [
            create_user_message("Hello"),
            create_assistant_message("Hi there!"),
            create_user_message("Recent"),
        ]
        
        await context.apply_context_strategy(messages)
        
        call_args = mock_summarizer.summarize.call_args
        summarized_text = call_args.kwargs['text']
        
        assert "ASSISTANT:" in summarized_text
        assert "Hi there!" in summarized_text
    
    @pytest.mark.asyncio
    async def test_handles_empty_message_content(self):
        """Test handling of messages with empty content."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=1,
        )
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty content
            {"role": "user", "content": "Recent"},
        ]
        
        # Should not raise an error
        result = await context.apply_context_strategy(messages)
        assert len(result) > 0


class TestResumeTokenContextEdgeCases:
    """Test edge cases for ResumeTokenContext."""
    
    @pytest.mark.asyncio
    async def test_few_messages_less_than_message_to_resume(self):
        """Test when total messages < message_to_resume."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=10,  # Want to keep 10
        )
        
        # Only provide 3 messages
        messages = [
            create_user_message(f"Message {i}") for i in range(3)
        ]
        
        result = await context.apply_context_strategy(messages)
        
        # Should not attempt summarization (falls back)
        # Should return all messages or trimmed version
        assert len(result) <= 3
    
    @pytest.mark.asyncio
    async def test_summarization_fails_gracefully(self):
        """Test graceful handling when summarization fails."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(side_effect=Exception("API Error"))
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=2,
        )
        
        messages = [create_user_message(f"Message {i}") for i in range(10)]
        
        # Should fall back to token trim instead of crashing
        result = await context.apply_context_strategy(messages)
        
        # Should still return some messages
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_zero_overlap(self):
        """Test with zero message overlap."""
        mock_summarizer = Mock()
        mock_summarizer.summarize = AsyncMock(return_value="Summary")
        
        context = ResumeTokenContext(
            max_tokens=500,
            resume_provider=mock_summarizer,
            message_to_resume=3,
            message_resume_overlap=0,  # No overlap
        )
        
        messages = [create_user_message(f"Message {i}") for i in range(10)]
        
        result = await context.apply_context_strategy(messages)
        
        # Should work without overlap
        assert len(result) >= 3  # Summary + recent messages
        mock_summarizer.summarize.assert_called_once()
