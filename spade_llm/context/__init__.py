"""SPADE_LLM context management module."""

from ._types import (  # Message type definitions; Helper functions for message creation
    AssistantMessage,
    ContextMessage,
    SummarizerProtocol,
    SystemMessage,
    ToolCall,
    ToolCallFunction,
    ToolResultMessage,
    UserMessage,
    create_assistant_message,
    create_assistant_tool_call_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
    spade_message_to_user_message,
)
from .context_manager import ContextManager
from .summarizer import Summarizer

from .management import (
 ContextManagement,
    NoContextManagement,
    SmartWindowSizeContext,
    WindowSizeContext,
    TokenBasedContext,  
    SmartTokenBasedContext,  
    AdaptiveTokenContext,
)

__all__ = [
    "ContextManager",
    # Context management strategies
    "ContextManagement",
    "NoContextManagement",
    "WindowSizeContext",
    "SmartWindowSizeContext",
    "TokenBasedContext",  
    "SmartTokenBasedContext",  
    "AdaptiveTokenContext",
    "ResumeTokenContext",

    # Summarizer
    "Summarizer",
    "SummarizerProtocol",

    # Message types
    "ContextMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "ToolCall",
    "ToolCallFunction",
    
    # Helper functions
    "create_system_message",
    "create_user_message",
    "spade_message_to_user_message",
    "create_assistant_message",
    "create_assistant_tool_call_message",
    "create_tool_result_message",
]
