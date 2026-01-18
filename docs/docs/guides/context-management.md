# Context Management

SPADE_LLM provides **context management** to control conversation memory and optimize LLM performance across multi-turn interactions.

## Overview

The **context management system** handles how conversation history is maintained and filtered:

- **Memory Control**: Prevents token overflow while preserving important information
- **Multi-Strategy Support**: Choose from basic, windowed, or extended strategies
- **Tool-Aware**: Maintains critical tool execution context
- **Conversation Isolation**: Each conversation maintains separate context

## Model-aware token counting

Most strategies can report **token usage** in `get_stats()`.

- Strategies accept an optional `model` parameter (e.g., `"gpt-4"`, `"gpt-4o-mini"`, `"claude-3-opus"`) used to count tokens via LiteLLM.
- **By default**, if you don't specify a `model`, the system automatically uses the model from the `LLMProvider` assigned to your `LLMAgent`. This ensures accurate token counting without manual configuration.
- You can override this by explicitly passing a `model` parameter to the context management strategy.
- If token counting fails for any reason, token fields may be reported as `0`.

## Context Strategies

### NoContextManagement (Default)

**Behavior**: Preserves all messages without any filtering or limitations.

```python
from spade_llm.context import NoContextManagement

# Keep all conversation history
context = NoContextManagement(model="gpt-4")

agent = LLMAgent(
    jid="assistant@example.com",
    password="password",
    provider=provider,
    context_management=context
)
```

**Characteristics**:
- ✅ Preserves complete conversation history
- ✅ No context loss
- ❌ Unlimited memory growth
- ❌ Potential LLM context overflow

**Use Cases**:
- Short conversations (< 10 exchanges)
- Debugging sessions requiring complete history
- Post-conversation analysis

### WindowSizeContext (Basic)

**Behavior**: Implements a sliding window maintaining only the last N messages.

```python
from spade_llm.context import WindowSizeContext

# Keep last 20 messages
context = WindowSizeContext(max_messages=20, model="gpt-4")

agent = LLMAgent(
    jid="assistant@example.com",
    password="password",
    provider=provider,
    context_management=context
)
```

**Visual Example**:

![Basic Window Context Management](../assets/images/window_context_managment_1.png)

*Basic sliding window keeps only the most recent N messages, dropping older ones as new messages arrive.*

**Characteristics**:
- ✅ Predictable memory control
- ✅ Prevents context overflow
- ❌ Loses important initial context
- ❌ No message type differentiation

**Use Cases**:
- Long conversations with memory constraints
- Resource-limited environments
- Continuous monitoring sessions

### SmartWindowSizeContext (Advanced) 🆕

**Behavior**: Management combining sliding window with selective retention of critical messages.

#### Basic Configuration

```python
from spade_llm.context import SmartWindowSizeContext

# Standard behavior (equivalent to WindowSizeContext)
basic_context = SmartWindowSizeContext(max_messages=20, model="gpt-4")

# With initial message preservation
initial_preserve = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    model="gpt-4"
)

# With tool prioritization
tool_priority = SmartWindowSizeContext(
    max_messages=20,
    prioritize_tools=True,
    model="gpt-4"
)

# Full configuration
smart_context = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    prioritize_tools=True,
    model="gpt-4"
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | `int` | 20 | Maximum messages in context |
| `preserve_initial` | `int` | 0 | Number of initial messages to always preserve |
| `prioritize_tools` | `bool` | False | Whether to prioritize tool results |
| `model` | `str` | `"gpt-3.5-turbo"` | Model name used for token counting |

#### Retention Algorithm

The SmartWindowSizeContext uses an algorithm:

1. **If** `total_messages ≤ max_messages` → Return all messages
2. **If** `preserve_initial = 0` and `prioritize_tools = False` → Basic sliding window
3. **If** `preserve_initial > 0` → Preserve initial messages + fill with recent ones
4. **If** `prioritize_tools = True` → Prioritize tool results + fill remaining space
5. **If** both enabled → Combine preservation + prioritization strategies

#### Advanced Features

##### Initial Message Preservation

```python
# Example: 30 total messages, window=10, preserve_initial=3
# Result: [msg1, msg2, msg3] + [msg24, msg25, ..., msg30]

context = SmartWindowSizeContext(
    max_messages=10, 
    preserve_initial=3
)
```

**Visual Example**:

![Smart Window with Initial Preservation](../assets/images/window_context_managment_2.png)

*Smart window preserves the first N messages (objectives, instructions) while filling remaining space with recent messages.*

**Benefits**: Preserves conversation objectives and fundamental context.

##### Tool Result Prioritization

```python
# Prioritizes all messages with role="tool"
context = SmartWindowSizeContext(
    max_messages=15, 
    prioritize_tools=True
)
```

**Algorithm**:
1. Extract all tool result messages
2. If tool messages ≥ max_messages → Keep recent tool messages
3. Otherwise → tool messages + recent messages to fill window
4. Reorder chronologically

##### Tool Call/Result Pair Preservation

The system automatically detects and preserves tool call/result pairs:

```python
# Automatically preserves:
# Assistant: "I'll check the weather" [tool_calls: get_weather]
# Tool: "22°C, sunny" [tool_call_id: matching_id]

context = SmartWindowSizeContext(
    max_messages=20,
    prioritize_tools=True
)
```

**Visual Example**:

![Smart Window with Tool Prioritization](../assets/images/window_context_managment_3.png)

*Smart window with initial preservation + tool prioritization maintains both conversation objectives and critical tool execution context.*

**Benefits**: Maintains execution context for complex tool workflows.

### TokenBasedContext (Token Budget) 🆕

**Behavior**: Keeps as many recent messages as possible within a **token budget**.

This is often more reliable than `max_messages` because message sizes vary.

```python
from spade_llm.context import TokenBasedContext

context = TokenBasedContext(
    max_tokens=4096,
    reserve_tokens=700,
    model="gpt-4"
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | 4096 | Maximum tokens allowed in context |
| `reserve_tokens` | `int` | 500 | Tokens reserved for the model response |
| `model` | `str` | `"gpt-3.5-turbo"` | Model name used for token counting |

### SmartTokenBasedContext (Token Budget + Tool-aware) 🆕

**Behavior**: Token-based version of `SmartWindowSizeContext`.

Combines:

- Token limiting
- Tool call/result pair preservation
- Optional initial message preservation
- Optional tool prioritization

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

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | 4096 | Maximum tokens allowed in context |
| `reserve_tokens` | `int` | 500 | Tokens reserved for the model response |
| `preserve_initial` | `int` | 0 | Number of initial messages to always preserve |
| `prioritize_tools` | `bool` | False | Whether to prioritize tool results |
| `model` | `str` | `"gpt-3.5-turbo"` | Model name used for token counting |

### AdaptiveTokenContext (Adaptive Selection) 🆕

**Behavior**: Scores messages by importance and selects a high-value subset under a target token utilization.

Use this when conversations are long and you want **automatic retention tradeoffs** without tuning message windows.

```python
from spade_llm.context import AdaptiveTokenContext

context = AdaptiveTokenContext(
    max_tokens=4096,
    reserve_tokens=500,
    target_utilization=0.85,
    model="gpt-4"
)
```

### ResumeTokenContext (Summary-based) 🆕

**Behavior**: Summarizes older messages using an LLM to compact the conversation while keeping recent messages in full detail.

This is ideal for **very long conversations** where you want to preserve context from early messages without storing them verbatim.

```python
from spade_llm.context import ResumeTokenContext, Summarizer
from spade_llm.providers import LLMProvider

# Create a provider for summarization (can be different from main provider)
summary_provider = LLMProvider(model="gpt-4o-mini")
summarizer = Summarizer(provider=summary_provider)

context = ResumeTokenContext(
    max_tokens=4096,
    reserve_tokens=500,
    resume_provider=summarizer,
    message_to_resume=4,  # Summarize when more than 4 old messages
    message_resume_overlap=1,  # Keep 1 message overlap for context
    model="gpt-4"
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | 4096 | Maximum tokens allowed in context |
| `reserve_tokens` | `int` | 500 | Tokens reserved for the model response |
| `resume_provider` | `SummarizerProtocol` | None | Summarizer instance for generating summaries |
| `message_to_resume` | `int` | 4 | Number of old messages before summarization |
| `message_resume_overlap` | `int` | 1 | Number of messages to keep as overlap between summary and recent messages |
| `summary_system_prompt` | `str` | None | Custom prompt for summarization (optional) |
| `model` | `str` | `"gpt-3.5-turbo"` | Model name used for token counting |

#### How it Works

1. **Keeps recent messages** in full detail (last N messages)
2. **Summarizes older messages** into a single compact message
3. **Preserves overlap** between summary and recent messages for continuity
4. **Respects token budget** by adjusting summary length

#### Example Workflow

```python
# With 20 messages and message_to_resume=4, overlap=1:
# Messages 1-15: Summarized into one message
# Message 15: Kept as overlap
# Messages 16-20: Kept in full detail
# Result: [Summary] + [msg15, msg16, msg17, msg18, msg19, msg20]
```

**Characteristics**:
- ✅ Preserves early conversation context through summarization
- ✅ Keeps recent messages in full detail
- ✅ Highly efficient for very long conversations
- ✅ Configurable overlap for context continuity
- ❌ Requires additional LLM call for summarization (adds latency)
- ❌ Summary may lose some nuanced details

**Use Cases**:
- Multi-hour customer support sessions
- Long-running research conversations
- Complex multi-step problem solving
- Conversations where early context remains relevant

**Visual Example**:

```
Original: [msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10]
                                    ↓
          With message_to_resume=4, overlap=1:
                                    ↓
Result:   [SUMMARY(msg1-6), msg6, msg7, msg8, msg9, msg10]
          └─ Summary ──┘  └─ Overlap + Recent messages ──┘
```

## Integration with LLMAgent

### Constructor Configuration

```python
from spade_llm.agent import LLMAgent
from spade_llm.context import SmartWindowSizeContext
from spade_llm.providers import LLMProvider

# Create provider
provider = LLMProvider(model="gpt-4")

# Create context strategy
# Note: model parameter is optional - by default uses provider's model
smart_context = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    prioritize_tools=True
    # model="gpt-4" <- Optional: defaults to provider's model automatically
)

# Integrate with agent
agent = LLMAgent(
    jid="agent@example.com",
    password="password",
    provider=provider,
    context_management=smart_context,
    system_prompt="You are an assistant with context management..."
)
```


## Context Statistics

### Getting Statistics

```python
context = SmartWindowSizeContext(
    max_messages=20,
    preserve_initial=3,
    prioritize_tools=True,
    model="gpt-4"
    model="gpt-4"
)

# Get stats for current conversation
stats = context.get_stats(total_messages=50)
```

### Statistics Format

```python
{
    "strategy": "smart_window_size",
    "max_messages": 20,
    "preserve_initial": 3,
    "prioritize_tools": True,
    "total_messages": 50,
    "messages_in_context": 20,
    "messages_dropped": 30,
    "model": "gpt-4",
    "tokens_before": 1234,
    "tokens_after": 890,
    "tokens_saved": 344,
    "compression_ratio": "72.12%"
}
```

## Strategy Comparison

### Message-count strategies

| Feature | NoContext | WindowSize | SmartWindowSize |
|---------|-----------|------------|-----------------|
| **Memory Control** | ❌ | ✅ | ✅ |
| **Preserves Initial Context** | ✅ | ❌ | ✅ (optional) |
| **Tool Prioritization** | ✅ | ❌ | ✅ (optional) |
| **Best for** | Short chats | Predictable windows | Tool-heavy workflows |
| **Memory Usage** | Unlimited | Limited | Limited |

### Token-budget strategies

| Feature | TokenBased | SmartTokenBased | AdaptiveToken | ResumeToken |
|---------|------------|-----------------|--------------|-------------|
| **Memory Control (tokens)** | ✅ | ✅ | ✅ | ✅ |
| **Preserves Initial Context** | ❌ | ✅ (optional) | ✅ (scored) | ✅ (via summary) |
| **Tool Pair Preservation** | ❌ | ✅ | ✅ (scored + pairs) | ✅ |
| **Summarization** | ❌ | ❌ | ❌ | ✅ |
| **Tuning effort** | Low | Medium | Low | Medium |
| **Best for** | Variable message sizes | Tool-heavy + strict budget | Long chats with mixed relevance | Very long conversations |


## Complete Example

### Example 1: Smart Window with Automatic Model Detection

```python
import asyncio
from spade_llm.agent import LLMAgent
from spade_llm.context import SmartWindowSizeContext
from spade_llm.providers import LLMProvider

async def main():
    # Create LLM provider
    provider = LLMProvider.create_openai(
        api_key="your-api-key",
        model="gpt-4"
    )
    
    # Configure context management
    # Model is automatically inherited from provider!
    smart_context = SmartWindowSizeContext(
        max_messages=20,
        preserve_initial=3,
        prioritize_tools=True
    )
    
    # Create agent with context management
    agent = LLMAgent(
        jid="smart_agent@example.com",
        password="password",
        provider=provider,
        context_management=smart_context,
        system_prompt="You are an assistant with context management."
    )
    
    await agent.start()
    
    # Monitor context during conversation
    while True:
        # ... agent processes messages ...
        
        # Check context stats periodically
        stats = agent.get_context_stats()
        if stats['messages_in_context'] > 15:
            print("Context approaching limit")
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Summary-based Context for Long Conversations

```python
import asyncio
from spade_llm.agent import LLMAgent
from spade_llm.context import ResumeTokenContext, Summarizer
from spade_llm.providers import LLMProvider

async def main():
    # Create main provider
    main_provider = LLMProvider(model="gpt-4")
    
    # Create separate provider for summarization (can use cheaper model)
    summary_provider = LLMProvider(model="gpt-4o-mini")
    summarizer = Summarizer(provider=summary_provider)
    
    # Configure summary-based context management
    resume_context = ResumeTokenContext(
        max_tokens=4096,
        reserve_tokens=500,
        resume_provider=summarizer,
        message_to_resume=4,
        message_resume_overlap=1
    )
    
    # Create agent
    agent = LLMAgent(
        jid="long_conversation_agent@example.com",
        password="password",
        provider=main_provider,
        context_management=resume_context,
        system_prompt="You are a helpful assistant for extended conversations."
    )
    
    await agent.start()
    print("Agent ready for long conversations with automatic summarization!")
    
    # Agent will automatically summarize old messages
    # while keeping recent ones in full detail

if __name__ == "__main__":
    asyncio.run(main())
```


## Next Steps

- **[Conversations](conversations.md)** - Learn about conversation lifecycle
- **[Memory System](memory.md)** - Explore agent memory capabilities
- **[Tools System](tools-system.md)** - Add tool capabilities
- **[API Reference](../reference/api/context.md)** - Detailed API documentation