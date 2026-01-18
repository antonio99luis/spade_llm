"""Context management strategies for controlling conversation history."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from ._types import ContextMessage, SummarizerProtocol
import litellm

logger = logging.getLogger("spade_llm.context.management")

try:
    import opentelemetry
    litellm.callbacks = ["langfuse_otel"]
    logger.info("LiteLLM tracing enabled with langfuse_otel callback")
except ImportError:
    logger.debug("OpenTelemetry not available, LiteLLM tracing disabled")


class ContextManagement(ABC):
    """Abstract base class for context management strategies."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize context management.
        
        Args:
            model: Model name for accurate token counting (e.g., 'gpt-4', 'claude-3-opus')
        """
        self.model = model or "gpt-3.5-turbo"
        self._last_token_count = 0
        self._last_applied_token_count = 0
        self._last_messages_before = []
        self._last_messages_after = []

    @abstractmethod
    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """
        Apply context management strategy to messages.

        Args:
            messages: List of conversation messages to manage
            system_prompt: Optional system prompt (for context)

        Returns:
            List of messages after applying the strategy
        """
        pass

    @abstractmethod
    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """
        Get statistics about context management.

        Args:
            total_messages: Total number of messages before management

        Returns:
            Dictionary with statistics about the strategy application
        """
        pass

    def count_tokens(self, messages: List[ContextMessage]) -> int:
        """
        Count tokens in messages using LiteLLM if available.
        
        Args:
            messages: List of messages to count tokens for
            
        Returns:
            Number of tokens (0 if LiteLLM not available)
        """
        try:
            # Filter out None or empty messages
            valid_messages = [
                msg for msg in messages 
                if msg and msg.get("content") is not None
            ]
            
            if not valid_messages:
                return 0
                
            token_count = litellm.token_counter(
                model=self.model,
                messages=valid_messages
            )
            return token_count
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return 0
    
    def _update_token_stats(
        self, 
        messages_before: List[ContextMessage],
        messages_after: List[ContextMessage]
    ) -> None:
        """
        Update internal token statistics.
        
        Args:
            messages_before: Messages before applying strategy
            messages_after: Messages after applying strategy
        """
        self._last_messages_before = messages_before
        self._last_messages_after = messages_after
        self._last_token_count = self.count_tokens(messages_before)
        self._last_applied_token_count = self.count_tokens(messages_after)


class NoContextManagement(ContextManagement):
    """Default context management strategy that keeps all messages."""

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Return all messages unchanged."""
        self._update_token_stats(messages, messages)
        return messages

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for no management strategy."""
        return {
            "strategy": "none",
            "total_messages": total_messages,
            "messages_in_context": total_messages,
            "messages_dropped": 0,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": 0,
        }


class WindowSizeContext(ContextManagement):
    """Window-based context management that keeps the last N messages."""

    def __init__(self, max_messages: int = 20, model: Optional[str] = None):
        """
        Initialize window size context management.

        Args:
            max_messages: Maximum number of messages to keep in context
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        
        super().__init__(model)
        self.max_messages = max_messages

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Keep only the last max_messages from the conversation."""
        if len(messages) <= self.max_messages:
            self._update_token_stats(messages, messages)
            return messages

        return messages[-self.max_messages:]

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for window size strategy."""
        messages_in_context = min(total_messages, self.max_messages)
        messages_dropped = max(0, total_messages - self.max_messages)
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)

        return {
            "strategy": "window_size",
            "max_messages": self.max_messages,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
        }


class SmartWindowSizeContext(ContextManagement):
    """Smart window-based context management with optional initial message preservation and tool prioritization."""

    def __init__(
        self,
        max_messages: int = 20,
        preserve_initial: int = 0,
        prioritize_tools: bool = False,
        model: Optional[str] = None
    ):
        """
        Initialize smart window size context management.

        Args:
            max_messages: Maximum number of messages to keep in context
            preserve_initial: Number of initial messages to always preserve (0 = disabled)
            prioritize_tools: Whether to prioritize tool results in selection
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        if preserve_initial < 0:
            raise ValueError("preserve_initial must be >= 0")
        if preserve_initial >= max_messages:
            raise ValueError("preserve_initial must be less than max_messages")

        self.max_messages = max_messages
        self.preserve_initial = preserve_initial
        self.prioritize_tools = prioritize_tools
        super().__init__(model)

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Apply smart context management strategy."""
        if len(messages) <= self.max_messages:
            self._update_token_stats(messages, messages)
            return messages

        if self.preserve_initial == 0 and not self.prioritize_tools:
            result = self._sliding_window_with_pairs(messages)

        elif self.preserve_initial > 0 and not self.prioritize_tools:
            result = self._preserve_initial_only(messages)

        elif self.preserve_initial == 0 and self.prioritize_tools:
            result = self._prioritize_tools_only(messages)
        
        else:
            result = self._smart_combination(messages)

        self._update_token_stats(messages, result)

        return result

    def _sliding_window_with_pairs(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Simple sliding window that preserves tool pairs."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set()
        current_pos = len(messages) - 1

        while len(selected_indices) < self.max_messages and current_pos >= 0:
            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices:
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                # Add all messages of the pair if we have space
                if len(selected_indices) + len(pair_for_msg) <= self.max_messages:
                    selected_indices.update(pair_for_msg)
                    current_pos = min(pair_for_msg) - 1
                else:
                    # Not enough space for the pair, skip it
                    current_pos = min(pair_for_msg) - 1
            else:
                # Regular message, add it
                selected_indices.add(current_pos)
                current_pos -= 1

        # Return messages in original order
        return [messages[i] for i in sorted(selected_indices)]

    def _preserve_initial_only(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Preserve initial N messages plus recent messages to fill window, respecting tool pairs."""
        initial = messages[: self.preserve_initial]
        remaining_space = self.max_messages - len(initial)

        if remaining_space <= 0:
            return initial[: self.max_messages]

        if len(messages) <= self.preserve_initial + remaining_space:
            return messages

        # Find tool pairs
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set(range(len(initial)))

        # Fill remaining space from the end, respecting tool pairs
        current_pos = len(messages) - 1
        while remaining_space > 0 and current_pos >= self.preserve_initial:
            if current_pos in selected_indices:
                current_pos -= 1
                continue

            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices:
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                # Only add if all messages in pair fit and are after initial messages
                pair_after_initial = [
                    idx for idx in pair_for_msg if idx >= self.preserve_initial
                ]
                if (len(pair_after_initial) == len(pair_for_msg)
                        and remaining_space >= len(pair_for_msg)
                        and not any(idx in selected_indices for idx in pair_for_msg)):
                    selected_indices.update(pair_for_msg)
                    remaining_space -= len(pair_for_msg)
                current_pos = min(pair_for_msg) - 1
            else:
                selected_indices.add(current_pos)
                remaining_space -= 1
                current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def _find_tool_pairs(self, messages: List[ContextMessage]) -> List[tuple]:
        """Find assistant-tool message pairs, handling multiple tool calls and results."""
        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            # Look for assistant messages with tool_calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
                call_ids = {call.get("id") for call in tool_calls if call.get("id")}

                # Find all corresponding tool result messages
                tool_result_indices = []
                j = i + 1

                # Look for tool results that match the call IDs
                while j < len(messages) and len(call_ids) > 0:
                    if messages[j].get("role") == "tool":
                        tool_call_id = messages[j].get("tool_call_id")
                        if tool_call_id in call_ids:
                            tool_result_indices.append(j)
                            call_ids.remove(tool_call_id)
                    elif messages[j].get("role") in ["user", "assistant"]:
                        # Stop looking if we hit another conversation turn
                        break
                    j += 1

                # Only create pairs if we found all tool results
                if len(call_ids) == 0 and tool_result_indices:
                    # Create a group that includes the assistant message and all its tool results
                    pair_indices = [i] + tool_result_indices
                    pairs.append(tuple(pair_indices))
                    i = max(tool_result_indices) + 1
                else:
                    i += 1
            else:
                i += 1
        return pairs

    def _prioritize_tools_only(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Prioritize tool results while preserving tool call/result pairs."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_indices = set()

        # First, add all tool pairs that fit (from most recent)
        for pair_indices in reversed(tool_pairs):
            if len(selected_indices) + len(pair_indices) <= self.max_messages:
                selected_indices.update(pair_indices)

        # Fill remaining space with non-tool messages
        remaining_space = self.max_messages - len(selected_indices)
        current_pos = len(messages) - 1

        while remaining_space > 0 and current_pos >= 0:
            # Skip messages that are part of tool pairs already selected
            is_in_pair = any(current_pos in pair_indices for pair_indices in tool_pairs)
            if current_pos not in selected_indices and not is_in_pair:
                selected_indices.add(current_pos)
                remaining_space -= 1
            current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def _smart_combination(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Combine initial preservation with tool prioritization while preserving tool pairs."""
        initial = messages[: self.preserve_initial]
        available_space = self.max_messages - len(initial)

        if available_space <= 0:
            return initial[: self.max_messages]

        # Find tool pairs in the entire conversation
        tool_pairs = self._find_tool_pairs(messages)

        # Filter pairs to only those after initial messages
        relevant_pairs = [
            pair_indices
            for pair_indices in tool_pairs
            if all(idx >= self.preserve_initial for idx in pair_indices)
        ]

        selected_indices = set(range(len(initial)))

        # Add tool pairs first (from most recent)
        for pair_indices in reversed(relevant_pairs):
            if available_space >= len(pair_indices):
                selected_indices.update(pair_indices)
                available_space -= len(pair_indices)

        # Fill remaining space with other messages from the end
        current_pos = len(messages) - 1
        while available_space > 0 and current_pos >= self.preserve_initial:
            # Skip messages that are part of tool pairs or already selected
            is_in_pair = any(
                current_pos in pair_indices for pair_indices in relevant_pairs
            )
            if current_pos not in selected_indices and not is_in_pair:
                selected_indices.add(current_pos)
                available_space -= 1
            current_pos -= 1

        return [messages[i] for i in sorted(selected_indices)]

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for smart window size strategy."""
        messages_in_context = min(total_messages, self.max_messages)
        messages_dropped = max(0, total_messages - self.max_messages)
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)
        compression_ratio = (
                self._last_applied_token_count / self._last_token_count 
                if self._last_token_count > 0 else 1.0
            )
        return {
            "strategy": "smart_window_size",
            "max_messages": self.max_messages,
            "preserve_initial": self.preserve_initial,
            "prioritize_tools": self.prioritize_tools,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
            "compression_ratio": f"{compression_ratio:.2%}",
        }

class TokenBasedContext(ContextManagement):
    """Token-based context management that keeps messages within a token limit."""

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 500,
        model: Optional[str] = None
    ):
        """
        Initialize token-based context management.

        Args:
            max_tokens: Maximum tokens allowed in context
            reserve_tokens: Tokens to reserve for model response
            model: Model name for accurate token counting
        """
        super().__init__(model)
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        if reserve_tokens >= max_tokens:
            raise ValueError("reserve_tokens must be less than max_tokens")

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Keep only messages that fit within token limit."""
        if not messages:
            self._update_token_stats([], [])
            return []

        # Count tokens for all messages
        total_tokens = self.count_tokens(messages)
        
        if total_tokens <= self.available_tokens:
            self._update_token_stats(messages, messages)
            return messages

        # Remove messages from the beginning until we fit
        result = []
        current_tokens = 0
        
        # Start from the end (most recent messages)
        for msg in reversed(messages):
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens <= self.available_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        self._update_token_stats(messages, result)
        return result

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for token-based strategy."""
        messages_in_context = len(self._last_messages_after) if self._last_messages_after else 0
        messages_dropped = total_messages - messages_in_context
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)
        
        return {
            "strategy": "token_based",
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.available_tokens,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
            "tokens_available": self.available_tokens - self._last_applied_token_count,
        }
    

class SmartTokenBasedContext(ContextManagement):
    """
    Smart token-based context management with tool pair preservation and initial message retention.
    
    This is the most sophisticated strategy, combining:
    - Token-based limiting (more accurate than message count)
    - Tool call/result pair preservation
    - Initial message preservation
    - Tool prioritization
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 500,
        preserve_initial: int = 0,
        prioritize_tools: bool = False,
        model: Optional[str] = None
    ):
        """
        Initialize smart token-based context management.

        Args:
            max_tokens: Maximum tokens allowed in context
            reserve_tokens: Tokens to reserve for model response
            preserve_initial: Number of initial messages to always preserve (0 = disabled)
            prioritize_tools: Whether to prioritize tool results in selection
            model: Model name for accurate token counting
        """
        super().__init__(model)
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        if reserve_tokens >= max_tokens:
            raise ValueError("reserve_tokens must be less than max_tokens")
        if preserve_initial < 0:
            raise ValueError("preserve_initial must be >= 0")

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens
        self.preserve_initial = preserve_initial
        self.prioritize_tools = prioritize_tools

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Apply smart token-based context management strategy."""
        if not messages:
            self._update_token_stats([], [])
            return []

        total_tokens = self.count_tokens(messages)
        
        if total_tokens <= self.available_tokens:
            self._update_token_stats(messages, messages)
            return messages

        # Strategy selection based on configuration
        if self.preserve_initial == 0 and not self.prioritize_tools:
            result = self._token_sliding_window_with_pairs(messages)
        elif self.preserve_initial > 0 and not self.prioritize_tools:
            result = self._token_preserve_initial_only(messages)
        elif self.preserve_initial == 0 and self.prioritize_tools:
            result = self._token_prioritize_tools_only(messages)
        else:
            result = self._token_smart_combination(messages)
        
        self._update_token_stats(messages, result)
        return result

    def _token_sliding_window_with_pairs(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Token-based sliding window that preserves tool pairs."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_messages = []
        current_tokens = 0
        current_pos = len(messages) - 1

        while current_tokens < self.available_tokens and current_pos >= 0:
            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices:
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                # Get all messages in the pair
                pair_messages = [messages[i] for i in pair_for_msg]
                pair_tokens = self.count_tokens(pair_messages)
                
                # Add pair if it fits
                if current_tokens + pair_tokens <= self.available_tokens:
                    selected_messages.extend(pair_messages)
                    current_tokens += pair_tokens
                
                current_pos = min(pair_for_msg) - 1
            else:
                # Regular message
                msg_tokens = self.count_tokens([messages[current_pos]])
                if current_tokens + msg_tokens <= self.available_tokens:
                    selected_messages.insert(0, messages[current_pos])
                    current_tokens += msg_tokens
                current_pos -= 1

        # Sort by original order
        return sorted(selected_messages, key=lambda m: messages.index(m))

    def _token_preserve_initial_only(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Preserve initial N messages plus recent messages to fill token limit."""
        if self.preserve_initial == 0:
            return self._token_sliding_window_with_pairs(messages)

        initial = messages[:self.preserve_initial]
        initial_tokens = self.count_tokens(initial)

        if initial_tokens >= self.available_tokens:
            # Initial messages alone exceed limit
            logger.warning(
                f"Initial {self.preserve_initial} messages ({initial_tokens} tokens) "
                f"exceed available tokens ({self.available_tokens})"
            )
            return initial

        remaining_tokens = self.available_tokens - initial_tokens
        tool_pairs = self._find_tool_pairs(messages)
        selected_messages = list(initial)
        current_tokens = initial_tokens
        current_pos = len(messages) - 1

        # Fill remaining space from the end, respecting tool pairs
        while current_tokens < self.available_tokens and current_pos >= self.preserve_initial:
            # Check if already selected
            if messages[current_pos] in selected_messages:
                current_pos -= 1
                continue

            # Check if part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if current_pos in pair_indices and all(idx >= self.preserve_initial for idx in pair_indices):
                    pair_for_msg = pair_indices
                    break

            if pair_for_msg:
                pair_messages = [messages[i] for i in pair_for_msg]
                pair_tokens = self.count_tokens(pair_messages)
                
                if current_tokens + pair_tokens <= self.available_tokens:
                    selected_messages.extend(pair_messages)
                    current_tokens += pair_tokens
                
                current_pos = min(pair_for_msg) - 1
            else:
                msg_tokens = self.count_tokens([messages[current_pos]])
                if current_tokens + msg_tokens <= self.available_tokens:
                    selected_messages.append(messages[current_pos])
                    current_tokens += msg_tokens
                current_pos -= 1

        return sorted(selected_messages, key=lambda m: messages.index(m))

    def _token_prioritize_tools_only(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Prioritize tool results while preserving tool call/result pairs (token-based)."""
        tool_pairs = self._find_tool_pairs(messages)
        selected_messages = []
        current_tokens = 0

        # First, add all tool pairs that fit (from most recent)
        for pair_indices in reversed(tool_pairs):
            pair_messages = [messages[i] for i in pair_indices]
            pair_tokens = self.count_tokens(pair_messages)
            
            if current_tokens + pair_tokens <= self.available_tokens:
                selected_messages.extend(pair_messages)
                current_tokens += pair_tokens

        # Fill remaining space with non-tool messages from the end
        current_pos = len(messages) - 1
        while current_tokens < self.available_tokens and current_pos >= 0:
            msg = messages[current_pos]
            
            # Skip if already selected or part of a tool pair
            is_in_pair = any(current_pos in pair_indices for pair_indices in tool_pairs)
            if msg not in selected_messages and not is_in_pair:
                msg_tokens = self.count_tokens([msg])
                if current_tokens + msg_tokens <= self.available_tokens:
                    selected_messages.append(msg)
                    current_tokens += msg_tokens
            
            current_pos -= 1

        return sorted(selected_messages, key=lambda m: messages.index(m))

    def _token_smart_combination(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Combine initial preservation with tool prioritization (token-based)."""
        if self.preserve_initial == 0:
            return self._token_prioritize_tools_only(messages)

        initial = messages[:self.preserve_initial]
        initial_tokens = self.count_tokens(initial)

        if initial_tokens >= self.available_tokens:
            logger.warning(
                f"Initial {self.preserve_initial} messages ({initial_tokens} tokens) "
                f"exceed available tokens ({self.available_tokens})"
            )
            return initial

        available_space = self.available_tokens - initial_tokens
        tool_pairs = self._find_tool_pairs(messages)
        
        # Filter pairs to only those after initial messages
        relevant_pairs = [
            pair_indices
            for pair_indices in tool_pairs
            if all(idx >= self.preserve_initial for idx in pair_indices)
        ]

        selected_messages = list(initial)
        current_tokens = initial_tokens

        # Add tool pairs first (from most recent)
        for pair_indices in reversed(relevant_pairs):
            pair_messages = [messages[i] for i in pair_indices]
            pair_tokens = self.count_tokens(pair_messages)
            
            if current_tokens + pair_tokens <= self.available_tokens:
                selected_messages.extend(pair_messages)
                current_tokens += pair_tokens

        # Fill remaining space with other messages from the end
        current_pos = len(messages) - 1
        while current_tokens < self.available_tokens and current_pos >= self.preserve_initial:
            msg = messages[current_pos]
            
            # Skip if already selected or part of a relevant tool pair
            is_in_pair = any(current_pos in pair_indices for pair_indices in relevant_pairs)
            if msg not in selected_messages and not is_in_pair:
                msg_tokens = self.count_tokens([msg])
                if current_tokens + msg_tokens <= self.available_tokens:
                    selected_messages.append(msg)
                    current_tokens += msg_tokens
            
            current_pos -= 1

        return sorted(selected_messages, key=lambda m: messages.index(m))

    def _find_tool_pairs(self, messages: List[ContextMessage]) -> List[tuple]:
        """Find assistant-tool message pairs, handling multiple tool calls and results."""
        # Reutilizar la lógica de SmartWindowSizeContext
        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
                call_ids = {call.get("id") for call in tool_calls if call.get("id")}

                tool_result_indices = []
                j = i + 1

                while j < len(messages) and len(call_ids) > 0:
                    if messages[j].get("role") == "tool":
                        tool_call_id = messages[j].get("tool_call_id")
                        if tool_call_id in call_ids:
                            tool_result_indices.append(j)
                            call_ids.remove(tool_call_id)
                    elif messages[j].get("role") in ["user", "assistant"]:
                        break
                    j += 1

                if len(call_ids) == 0 and tool_result_indices:
                    pair_indices = [i] + tool_result_indices
                    pairs.append(tuple(pair_indices))
                    i = max(tool_result_indices) + 1
                else:
                    i += 1
            else:
                i += 1
        return pairs

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for smart token-based strategy."""
        messages_in_context = len(self._last_messages_after) if self._last_messages_after else 0
        messages_dropped = total_messages - messages_in_context
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)
        compression_ratio = (
            self._last_applied_token_count / self._last_token_count 
            if self._last_token_count > 0 else 1.0
        )

        return {
            "strategy": "smart_token_based",
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.available_tokens,
            "preserve_initial": self.preserve_initial,
            "prioritize_tools": self.prioritize_tools,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
            "tokens_available": self.available_tokens - self._last_applied_token_count,
            "compression_ratio": f"{compression_ratio:.2%}",
        }

#TODO: Comprobar eficiencia respecto a SmartTokenBasedContext     
class AdaptiveTokenContext(ContextManagement):
    """
    Adaptive context management that automatically adjusts based on token usage patterns.
    
    Features:
    - Monitors token usage over time
    - Dynamically adjusts retention strategy
    - Learns from conversation patterns
    - Optimizes for both recency and relevance
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 500,
        target_utilization: float = 0.85,
        model: Optional[str] = None
    ):
        """
        Initialize adaptive token-based context management.

        Args:
            max_tokens: Maximum tokens allowed in context
            reserve_tokens: Tokens to reserve for model response
            target_utilization: Target token utilization (0.0-1.0)
            model: Model name for accurate token counting
        """
        super().__init__(model)
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError("target_utilization must be between 0.0 and 1.0")

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens
        self.target_utilization = target_utilization
        self.target_tokens = int(self.available_tokens * target_utilization)
        
        # Tracking for adaptive behavior
        self._token_history = []
        self._message_importance_scores = {}

    def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Apply adaptive token-based context management."""
        if not messages:
            self._update_token_stats([], [])
            return []

        total_tokens = self.count_tokens(messages)
        
        if total_tokens <= self.target_tokens:
            self._update_token_stats(messages, messages)
            self._track_usage(total_tokens)
            return messages

        # Score messages by importance
        scored_messages = self._score_messages(messages)
        
        # Select messages to keep based on scores and token budget
        result = self._select_messages_by_score(scored_messages, messages)
        
        self._update_token_stats(messages, result)
        self._track_usage(self.count_tokens(result))
        return result

    def _score_messages(
        self, messages: List[ContextMessage]
    ) -> Dict[int, float]:
        """
        Score messages by importance.
        
        Scoring criteria:
        - Recency (higher score for recent messages)
        - Role (tool results scored higher)
        - Length (shorter messages may be more important)
        - Position (first few messages scored higher)
        """
        scores = {}
        total_messages = len(messages)
        
        for i, msg in enumerate(messages):
            score = 0.0
            
            # Recency score (0.0 to 1.0, higher for recent)
            recency = i / total_messages if total_messages > 1 else 1.0
            score += recency * 0.4
            
            # Role score
            role = msg.get("role", "")
            if role == "tool":
                score += 0.3  # Tool results are important
            elif role == "assistant":
                score += 0.2
            elif role == "user":
                score += 0.15
            
            # Initial message bonus
            if i < 3:
                score += 0.2
            
            # Tool pair membership bonus
            if msg.get("tool_calls") or msg.get("tool_call_id"):
                score += 0.15
            
            scores[i] = score
        
        return scores

    def _select_messages_by_score(
        self, 
        scores: Dict[int, float], 
        messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Select messages to keep based on scores and token budget."""
        # Sort message indices by score (descending)
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        
        selected_messages = []
        current_tokens = 0
        tool_pairs = self._find_tool_pairs(messages)
        
        # Build a set of already selected indices
        selected_indices = set()
        
        for idx in sorted_indices:
            if idx in selected_indices:
                continue
            
            # Check if this message is part of a tool pair
            pair_for_msg = None
            for pair_indices in tool_pairs:
                if idx in pair_indices:
                    pair_for_msg = pair_indices
                    break
            
            if pair_for_msg:
                # Add entire pair
                pair_messages = [messages[i] for i in pair_for_msg]
                pair_tokens = self.count_tokens(pair_messages)
                
                if current_tokens + pair_tokens <= self.target_tokens:
                    selected_messages.extend(pair_messages)
                    selected_indices.update(pair_for_msg)
                    current_tokens += pair_tokens
            else:
                # Add single message
                msg_tokens = self.count_tokens([messages[idx]])
                if current_tokens + msg_tokens <= self.target_tokens:
                    selected_messages.append(messages[idx])
                    selected_indices.add(idx)
                    current_tokens += msg_tokens
        
        # Sort by original order
        return sorted(selected_messages, key=lambda m: messages.index(m))

    def _find_tool_pairs(self, messages: List[ContextMessage]) -> List[tuple]:
        """Find assistant-tool message pairs."""
        # Reutilizar la lógica de SmartTokenBasedContext
        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg.get("tool_calls", [])
                call_ids = {call.get("id") for call in tool_calls if call.get("id")}

                tool_result_indices = []
                j = i + 1

                while j < len(messages) and len(call_ids) > 0:
                    if messages[j].get("role") == "tool":
                        tool_call_id = messages[j].get("tool_call_id")
                        if tool_call_id in call_ids:
                            tool_result_indices.append(j)
                            call_ids.remove(tool_call_id)
                    elif messages[j].get("role") in ["user", "assistant"]:
                        break
                    j += 1

                if len(call_ids) == 0 and tool_result_indices:
                    pair_indices = [i] + tool_result_indices
                    pairs.append(tuple(pair_indices))
                    i = max(tool_result_indices) + 1
                else:
                    i += 1
            else:
                i += 1
        return pairs

    def _track_usage(self, tokens_used: int) -> None:
        """Track token usage for adaptive behavior."""
        self._token_history.append(tokens_used)
        # Keep only last 100 measurements
        if len(self._token_history) > 100:
            self._token_history.pop(0)

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for adaptive token-based strategy."""
        messages_in_context = len(self._last_messages_after) if self._last_messages_after else 0
        messages_dropped = total_messages - messages_in_context
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)
        
        # Calculate average token usage
        avg_token_usage = (
            sum(self._token_history) / len(self._token_history)
            if self._token_history else 0
        )
        
        return {
            "strategy": "adaptive_token",
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.available_tokens,
            "target_tokens": self.target_tokens,
            "target_utilization": f"{self.target_utilization:.2%}",
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
            "avg_token_usage": int(avg_token_usage),
            "token_history_samples": len(self._token_history),
        }
    
class ResumeTokenContext(ContextManagement):
    """Summary-based context management that compacts older messages via a summarizer.

    This strategy:
    - Keeps the most recent messages in full detail.
    - Summarizes older messages into a single synthetic message using a provider.
    - Ensures the final context fits within a token budget.
    """
    DEFAULT_SUMMARY_PROMPT = (
        "You are a conversation summarizer. Summarize the following conversation "
        "keeping the most important information, context, and key decisions. "
        "Be concise but preserve critical details."
    )
    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 500,
        model: Optional[str] = None,
        resume_provider: Optional[SummarizerProtocol] = None,
        message_to_resume: int = 4,
        message_resume_overlap: int = 1,
        summary_system_prompt: Optional[str] = None,
    ):
        """
        Initialize summary-based token context management.

        Args:
            max_tokens: Maximum tokens allowed in context (including summary).
            reserve_tokens: Tokens to reserve for model response.
            model: Model name for accurate token counting.
            resume_provider: Summarizer instance used to summarize old messages.
            message_to_resume: Number of most recent messages to keep un-summarized
                               (equivalent to compaction_interval).
            message_resume_overlap: Number of overlapping recent messages that
                                    appear both in the summary and as full
                                    messages (equivalent to overlap_size).
            summary_system_prompt: Optional system prompt to guide the summarization.
        """
        super().__init__(model)

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        if reserve_tokens >= max_tokens:
            raise ValueError("reserve_tokens must be less than max_tokens")
        if message_to_resume <= 0:
            raise ValueError("message_to_resume must be greater than 0")
        if message_resume_overlap < 0:
            raise ValueError("message_resume_overlap must be >= 0")

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

        self.resume_provider = resume_provider
        self.message_to_resume = message_to_resume
        self.message_resume_overlap = message_resume_overlap
        self.summary_system_prompt = summary_system_prompt or self.DEFAULT_SUMMARY_PROMPT
        # Stats/adicional
        self._last_used_summary = False
        self._last_summary_tokens = 0
        self._last_original_messages = 0

    async def apply_context_strategy(
        self, messages: List[ContextMessage], system_prompt: Optional[str] = None
    ) -> List[ContextMessage]:
        """Apply summary-based context management strategy."""
        self._last_used_summary = False
        self._last_summary_tokens = 0
        self._last_original_messages = len(messages)

        if not messages:
            self._update_token_stats([], [])
            return []

        total_tokens = self.count_tokens(messages)
        if total_tokens <= self.available_tokens:
            self._update_token_stats(messages, messages)
            return messages

        # Si no hay proveedor de resumen o muy pocos mensajes, fallback a recorte simple
        if self.resume_provider is None or len(messages) <= self.message_to_resume:
            logger.info(
                f"No resume provider or too few messages ({len(messages)} <= {self.message_to_resume}), "
                "using fallback token trim"
            )
            result = self._fallback_token_trim(messages)
            self._update_token_stats(messages, result)
            return result

        # Dividir en bloque a resumir (antiguo) y bloque reciente
        # Los últimos 'message_to_resume' mensajes se mantienen completos
        split_index = max(0, len(messages) - self.message_to_resume)
        
        # Mensajes antiguos que se van a resumir
        old_messages = messages[:split_index]
        
        # Mensajes recientes que se mantienen completos
        recent_messages = messages[split_index:]
        
        # Si hay overlap, incluir los primeros N mensajes recientes en el resumen también
        if self.message_resume_overlap > 0 and len(recent_messages) > 0:
            overlap_count = min(self.message_resume_overlap, len(recent_messages))
            messages_to_summarize = old_messages + recent_messages[:overlap_count]
            logger.info(
                f"Summarizing {len(old_messages)} old + {overlap_count} overlap = "
                f"{len(messages_to_summarize)} messages, keeping {len(recent_messages)} full"
            )
        else:
            messages_to_summarize = old_messages
            logger.info(
                f"Summarizing {len(old_messages)} messages, keeping {len(recent_messages)} full"
            )

        summary_text = await self._summarize_messages(messages_to_summarize, system_prompt)
        if not summary_text:
            # Si el resumen falla, usar recorte simple
            result = self._fallback_token_trim(messages)
            self._update_token_stats(messages, result)
            return result

        summary_message: ContextMessage = {
            "role": "assistant",
            "content": summary_text,
        }

        candidate = [summary_message] + recent_messages
        candidate_tokens = self.count_tokens(candidate)

        if candidate_tokens > self.available_tokens:
            # Aún excede el límite: reducir mensajes recientes pero MANTENER el resumen
            summary_tokens = self.count_tokens([summary_message])
            remaining_tokens = self.available_tokens - summary_tokens
            
            # Si el resumen solo ya excede el límite, usamos fallback completo
            if remaining_tokens <= 0:
                logger.warning(
                    f"Summary alone ({summary_tokens} tokens) exceeds available tokens "
                    f"({self.available_tokens}). Using fallback trim."
                )
                result = self._fallback_token_trim(messages)
            else:
                # Recortar mensajes recientes para que quepan con el resumen
                trimmed_recent = []
                current_tokens = 0
                for msg in reversed(recent_messages):
                    msg_tokens = self.count_tokens([msg])
                    if current_tokens + msg_tokens <= remaining_tokens:
                        trimmed_recent.insert(0, msg)
                        current_tokens += msg_tokens
                    else:
                        break
                
                result = [summary_message] + trimmed_recent
                logger.info(
                    f"Summary + {len(trimmed_recent)}/{len(recent_messages)} recent messages "
                    f"fit in {self.available_tokens} tokens"
                )
        else:
            result = candidate

        # Actualizar estadísticas de resumen
        self._last_used_summary = True
        self._last_summary_tokens = self.count_tokens([summary_message])

        self._update_token_stats(messages, result)
        return result

    def _fallback_token_trim(
        self, messages: List[ContextMessage]
    ) -> List[ContextMessage]:
        """Simple token-based trimming from the beginning (most recent first)."""
        result: List[ContextMessage] = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens <= self.available_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        return result

    async def _summarize_messages(
        self,
        messages: List[ContextMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build a textual representation of messages and call the resume provider."""
        if not messages or self.resume_provider is None:
            return ""

        # Construir texto de entrada para el resumidor
        lines: List[str] = []
        if system_prompt:
            lines.append(f"SYSTEM: {system_prompt}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if not content:
                continue

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                line = f"TOOL[{tool_id}]: {content}"
            else:
                line = f"{role.upper()}: {content}"
            lines.append(line)

        summary_input = "\n".join(lines)

        # Build complete prompt with context
        complete_prompt = f"{self.summary_system_prompt}\n\nContext: {summary_input}"

        try:
            provider = self.resume_provider
            # Usar el método summarize del provider
            if hasattr(provider, "summarize"):
                # Preparar metadata para tracing
                metadata = {
                    "session_id": "context_summary",
                    "tags": ["context_management", "summarization"]
                }
                
                summary = await provider.summarize(
                    text=summary_input,
                    prompt=complete_prompt,
                    metadata=metadata
                )
                logger.info(f"Generated summary of {len(messages)} messages: {summary[:100]}...")
                return summary
            elif callable(provider):
                # Fallback para callables personalizados
                return provider(
                    summary_input,
                    system_prompt=complete_prompt,
                    max_tokens=self.available_tokens,
                )
            else:
                logger.warning(
                    "resume_provider must be callable or expose a 'summarize' method"
                )
                return ""
        except Exception as e:
            logger.warning(f"Error while summarizing context: {e}", exc_info=True)
            return ""

    def get_stats(self, total_messages: int) -> Dict[str, Any]:
        """Return statistics for summary-based token strategy."""
        messages_in_context = (
            len(self._last_messages_after) if self._last_messages_after else 0
        )
        messages_dropped = total_messages - messages_in_context
        tokens_saved = max(0, self._last_token_count - self._last_applied_token_count)

        return {
            "strategy": "resume_token_based",
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.available_tokens,
            "message_to_resume": self.message_to_resume,
            "message_resume_overlap": self.message_resume_overlap,
            "used_summary": self._last_used_summary,
            "summary_tokens": self._last_summary_tokens,
            "total_messages": total_messages,
            "messages_in_context": messages_in_context,
            "messages_dropped": messages_dropped,
            "model": self.model,
            "tokens_before": self._last_token_count,
            "tokens_after": self._last_applied_token_count,
            "tokens_saved": tokens_saved,
            "tokens_available": self.available_tokens - self._last_applied_token_count,
        }