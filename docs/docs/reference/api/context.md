# Context API

API reference for conversation context management and strategies.

## ContextManager

Manages conversation history and context for LLM interactions.

### Constructor

```python
ContextManager(
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
    context_management: Optional[ContextManagement] = None,
    model: Optional[str] = None,
    provider: Optional[BaseLLMProvider] = None
)
```

**Parameters:**

- `max_tokens` - Maximum context size in tokens
- `system_prompt` - System instructions for LLM
- `context_management` - Context management strategy (defaults to NoContextManagement)
- `model` - Model name for token counting (optional, see below)
- `provider` - LLM provider instance (optional, used to auto-detect model)

**Model Resolution:**

The model used for token counting is resolved with the following priority:

1. **Explicit `model` parameter** - If provided, uses this model
2. **Provider's model** - If `provider` is given and has a `model` attribute, uses that
3. **Fallback** - Defaults to `"gpt-3.5-turbo"` if neither is available

This automatic resolution ensures accurate token counting without manual configuration when using with `LLMAgent`.

### Methods

#### add_message()

```python
def add_message(self, message: Message, conversation_id: str) -> None
```

Add SPADE message to conversation context.

**Example:**

```python
context = ContextManager(system_prompt="You are helpful")
context.add_message(spade_message, "user1_session")
```

#### add_message_dict()

```python
def add_message_dict(self, message_dict: ContextMessage, conversation_id: str) -> None
```

Add message from dictionary format.

**Example:**

```python
user_msg = {"role": "user", "content": "Hello!"}
context.add_message_dict(user_msg, "user1_session")
```

#### add_assistant_message()

```python
def add_assistant_message(self, content: str, conversation_id: Optional[str] = None) -> None
```

Add assistant response to context.

**Example:**

```python
context.add_assistant_message("Hello! How can I help?", "user1_session")
```

#### add_tool_result()

```python
def add_tool_result(
    self, 
    tool_name: str, 
    result: Any, 
    tool_call_id: str, 
    conversation_id: Optional[str] = None
) -> None
```

Add tool execution result to context.

**Example:**

```python
context.add_tool_result(
    tool_name="get_weather",
    result="22°C, sunny",
    tool_call_id="call_123",
    conversation_id="user1_session"
)
```

#### get_prompt()

```python
def get_prompt(self, conversation_id: Optional[str] = None) -> List[ContextMessage]
```

Get formatted prompt for LLM provider.

**Example:**

```python
prompt = context.get_prompt("user1_session")
# Returns list of messages formatted for LLM
```

#### get_conversation_history()

```python
def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[ContextMessage]
```

Get raw conversation history.

**Example:**

```python
history = context.get_conversation_history("user1_session")
print(f"Conversation has {len(history)} messages")
```

#### clear()

```python
def clear(self, conversation_id: Optional[str] = None) -> None
```

Clear conversation messages.

**Example:**

```python
# Clear specific conversation
context.clear("user1_session")

# Clear all conversations
context.clear("all")
```

#### get_active_conversations()

```python
def get_active_conversations(self) -> List[str]
```

Get list of active conversation IDs.

**Example:**

```python
conversations = context.get_active_conversations()
print(f"Active conversations: {conversations}")
```

#### set_current_conversation()

```python
def set_current_conversation(self, conversation_id: str) -> bool
```

Set current conversation context.

**Example:**

```python
success = context.set_current_conversation("user1_session")
```

### Example Usage

```python
from spade_llm.context import ContextManager

# Create context manager
context = ContextManager(
    system_prompt="You are a helpful coding assistant",
    max_tokens=2000
)

# Add conversation messages
context.add_message_dict(
    {"role": "user", "content": "Help me with Python"}, 
    "coding_session"
)

context.add_assistant_message(
    "I'd be happy to help with Python!", 
    "coding_session"
)

# Get formatted prompt
prompt = context.get_prompt("coding_session")
# Use with LLM provider
```

## CoordinationContextManager

Context manager tailored for coordination scenarios where a coordinator agent supervises several subagents.

```python
CoordinationContextManager(
    coordination_session: str,
    subagent_ids: Set[str],
    **kwargs
)
```

**Additional behaviour:**

- Forces every message involving a registered subagent to use the shared `coordination_session` thread.
- Preserves standard handling for external participants (messages outside the organization keep their original thread).
- Provides `add_coordination_command(target_agent: str, command: str)` to record coordinator-issued instructions inside the shared context.

**Usage Example:**

```python
from spade_llm.agent.coordinator_agent import CoordinationContextManager

coordination_context = CoordinationContextManager(
    coordination_session="city_ops",
    subagent_ids={"traffic@xmpp.local", "alerts@xmpp.local"}
)

# Attach to CoordinatorAgent via `_context_override` or constructor kwargs
```

## Context Management Strategies

### ContextManagement (Abstract Base)

Base class for all context management strategies.

```python
from spade_llm.context.management import ContextManagement
```

#### Constructor

All built-in strategies accept an optional `model` parameter, used for more accurate token counting.

```python
ContextManagement(model: Optional[str] = None)
```

**Model Parameter Behavior:**

- If `model` is explicitly provided, uses that model for token counting
- If `model` is omitted and the context management is used with an `LLMAgent`, the agent's provider model is automatically used
- If neither is available, defaults to `"gpt-3.5-turbo"`

**Best Practice:** When creating an `LLMAgent`, you typically don't need to specify the `model` parameter in context strategies - it will be automatically inherited from your provider for accurate token counting.

#### apply_context_strategy()

```python
def apply_context_strategy(
    self, 
    messages: List[ContextMessage], 
    system_prompt: Optional[str] = None
) -> List[ContextMessage]
```

Apply the context management strategy to messages.

#### get_stats()

```python
def get_stats(self, total_messages: int) -> Dict[str, Any]
```

Get statistics about context management.

#### count_tokens()

Token counting is implemented via LiteLLM (`litellm.token_counter`) and depends on the configured `model`.

```python
def count_tokens(self, messages: List[ContextMessage]) -> int
```

Notes:

- If token counting fails for any reason, it returns `0` and logs a warning.
- Most strategies include token-related fields in `get_stats()` such as `tokens_before`, `tokens_after`, and `tokens_saved`.

### NoContextManagement

Preserves all messages without any filtering.

```python
from spade_llm.context import NoContextManagement

context = NoContextManagement()
```

**Optional parameters:**

- `model` (str, optional): Model name used for token counting

### WindowSizeContext

Basic sliding window context management.

```python
from spade_llm.context import WindowSizeContext

context = WindowSizeContext(max_messages=20)
```

**Parameters:**
- `max_messages` (int): Maximum number of messages to keep
- `model` (str, optional): Model name used for token counting

### SmartWindowSizeContext

Context management with message selection.

```python
from spade_llm.context import SmartWindowSizeContext

context = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    prioritize_tools=True,
    model="gpt-4"
)
```

**Parameters:**
- `max_messages` (int): Maximum number of messages to keep
- `preserve_initial` (int, optional): Number of initial messages to always preserve
- `prioritize_tools` (bool, optional): Whether to prioritize tool result messages
- `model` (str, optional): Model name used for token counting

**Example:**

```python
# Smart context with tool prioritization
smart_context = SmartWindowSizeContext(
    max_messages=25,
    preserve_initial=3,
    prioritize_tools=True
)

# Get statistics
stats = smart_context.get_stats(total_messages=50)
# Returns: {"strategy": "smart_window_size", "max_messages": 25, ...}
```

### TokenBasedContext

Token-budget context management that keeps only messages that fit within a token limit.

```python
from spade_llm.context import TokenBasedContext

context = TokenBasedContext(
    max_tokens=4096,
    reserve_tokens=500,
    model="gpt-4"
)
```

**Parameters:**

- `max_tokens` (int): Maximum tokens allowed in context
- `reserve_tokens` (int): Tokens reserved for the model response
- `model` (str, optional): Model name used for token counting

### SmartTokenBasedContext

Token-budget strategy with optional initial preservation and tool-aware retention.

```python
from spade_llm.context import SmartTokenBasedContext

context = SmartTokenBasedContext(
    max_tokens=4096,
    reserve_tokens=700,
    preserve_initial=3,
    prioritize_tools=True,
    model="gpt-4"
)
```

**Parameters:**

- `max_tokens` (int): Maximum tokens allowed in context
- `reserve_tokens` (int): Tokens reserved for the model response
- `preserve_initial` (int): Number of initial messages to always preserve
- `prioritize_tools` (bool): Whether to prioritize tool call/result context
- `model` (str, optional): Model name used for token counting

### AdaptiveTokenContext

Adaptive token-budget strategy that scores messages and retains a high-value subset.

```python
from spade_llm.context import AdaptiveTokenContext

context = AdaptiveTokenContext(
    max_tokens=4096,
    reserve_tokens=500,
    target_utilization=0.85,
    model="gpt-4"
)
```

**Parameters:**

- `max_tokens` (int): Maximum tokens allowed in context
- `reserve_tokens` (int): Tokens reserved for the model response
- `target_utilization` (float): Target utilization of the available token budget (0.0-1.0)
- `model` (str, optional): Model name used for token counting

### ResumeTokenContext

Summary-based context management that summarizes older messages using an LLM while keeping recent messages in full detail.

```python
from spade_llm.context import ResumeTokenContext, Summarizer
from spade_llm.providers import LLMProvider

# Create summarizer
summary_provider = LLMProvider(model="gpt-4o-mini")
summarizer = Summarizer(provider=summary_provider)

context = ResumeTokenContext(
    max_tokens=4096,
    reserve_tokens=500,
    resume_provider=summarizer,
    message_to_resume=4,
    message_resume_overlap=1,
    summary_system_prompt="Custom summary prompt...",  # Optional
    model="gpt-4"
)
```

**Parameters:**

- `max_tokens` (int): Maximum tokens allowed in context
- `reserve_tokens` (int): Tokens reserved for the model response
- `resume_provider` (SummarizerProtocol): Summarizer instance for generating summaries
- `message_to_resume` (int): Number of old messages before summarization is triggered
- `message_resume_overlap` (int): Number of messages to keep as overlap between summary and recent messages
- `summary_system_prompt` (str, optional): Custom prompt for summarization
- `model` (str, optional): Model name used for token counting

**Note:** This strategy's `apply_context_strategy` method is **async** and must be awaited.

## Message Types

### ContextMessage Types

```python
from spade_llm.context._types import (
    SystemMessage,
    UserMessage, 
    AssistantMessage,
    ToolResultMessage
)
```

#### SystemMessage

```python
{
    "role": "system",
    "content": "You are a helpful assistant"
}
```

#### UserMessage

```python
{
    "role": "user",
    "content": "Hello, how are you?",
    "name": "user@example.com"  # Optional
}
```

#### AssistantMessage

```python
# Text response
{
    "role": "assistant",
    "content": "I'm doing well, thank you!"
}

# With tool calls
{
    "role": "assistant", 
    "content": None,
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"Madrid\"}"
            }
        }
    ]
}
```

#### ToolResultMessage

```python
{
    "role": "tool",
    "content": "Weather in Madrid: 22°C, sunny",
    "tool_call_id": "call_123"
}
```


