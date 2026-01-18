"""
Context Management Strategies Experiment

Demonstrates different context management strategies with LLMAgent.
For each strategy, we send 5+ messages to the agent and display
the context that remains after applying each strategy.

PREREQUISITES:
1. Start SPADE built-in server in another terminal:
   spade run
   
2. Install dependencies:
   pip install spade_llm

This example uses SPADE's default built-in server (localhost:5222) - no account registration needed!
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import spade
from spade.message import Message

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.context import ContextManager, Summarizer
from spade_llm.context.management import (
    AdaptiveTokenContext,
    NoContextManagement,
    ResumeTokenContext,
    SmartTokenBasedContext,
    SmartWindowSizeContext,
    TokenBasedContext,
    WindowSizeContext,
)
from spade_llm.providers import LLMProvider
from spade_llm.utils import load_env_vars


# Configuration (will be loaded from .env file)
MAX_CONTEXT_TOKENS = 1000

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about literature and history.
Keep your responses concise but informative.
"""

# Test messages to send to the agent
TEST_MESSAGES = [
    "Hola, ¿quién fue Don Quijote de la Mancha?",
    "¿Cuál era el nombre real de Don Quijote?",
    "¿Quién era Sancho Panza y cuál era su rol?",
    "¿En qué época se escribió la novela?",
    "¿Cuál es la frase más famosa de Don Quijote?",
    "¿Qué simboliza Dulcinea en la obra?",
    "¿Cómo termina la historia de Don Quijote?",
]


@dataclass(frozen=True)
class StrategySpec:
    """Specification for a context management strategy."""
    name: str
    context_management: object
    description: str


def build_strategies(resume_model: str) -> List[StrategySpec]:
    """Build all context management strategies for testing."""
    
    # Create a summarizer for strategies that need it
    resume_provider = LLMProvider(model=resume_model)
    summarizer = Summarizer(provider=resume_provider)
    
    return [
        StrategySpec(
            "NoContextManagement",
            NoContextManagement(),
            "Keeps all messages without any pruning"
        ),
        StrategySpec(
            "WindowSizeContext (max=4)",
            WindowSizeContext(max_messages=4),
            "Keeps only the last N messages"
        ),
        StrategySpec(
            "SmartWindowSizeContext (max=4, preserve=1)",
            SmartWindowSizeContext(
                max_messages=4, 
                preserve_initial=1, 
                prioritize_tools=False
            ),
            "Keeps last N messages but preserves the first message"
        ),
        StrategySpec(
            f"TokenBasedContext (max={MAX_CONTEXT_TOKENS})",
            TokenBasedContext(
                max_tokens=MAX_CONTEXT_TOKENS, 
                reserve_tokens=100
            ),
            "Prunes oldest messages when token limit exceeded"
        ),
        StrategySpec(
            f"SmartTokenBasedContext (max={MAX_CONTEXT_TOKENS}, preserve=1)",
            SmartTokenBasedContext(
                max_tokens=MAX_CONTEXT_TOKENS,
                reserve_tokens=100,
                preserve_initial=1,
                prioritize_tools=False
            ),
            "Token-based pruning but preserves initial messages"
        ),
        StrategySpec(
            f"AdaptiveTokenContext (max={MAX_CONTEXT_TOKENS})",
            AdaptiveTokenContext(
                max_tokens=MAX_CONTEXT_TOKENS, 
                reserve_tokens=100, 
                target_utilization=0.85
            ),
            "Dynamically adjusts context based on token utilization"
        ),
        StrategySpec(
            f"ResumeTokenContext (max={MAX_CONTEXT_TOKENS}, resume=4, overlap=1)",
            ResumeTokenContext(
                max_tokens=MAX_CONTEXT_TOKENS,
                reserve_tokens=100,
                resume_provider=summarizer,  # Use the Summarizer instance
                message_to_resume=4,
                message_resume_overlap=1
            ),
            "Summarizes old messages and keeps recent ones in full detail"
        ),
    ]


def format_context_preview(messages: Sequence[Dict[str, str]], max_chars: int = 400) -> str:
    """Format a preview of the context messages."""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        # Replace newlines for cleaner display
        content = content.replace("\n", " ")
        lines.append(f"  [{i+1}] {role.upper()}: {content}")
    return "\n".join(lines)


def _sum_chars(messages: Sequence[Dict[str, str]]) -> int:
    """Compute total character length across message contents."""
    total = 0
    for msg in messages:
        content = (msg.get("content") or "")
        total += len(content)
    return total


def print_strategy_report(
    spec: StrategySpec,
    original_messages: List[Dict[str, str]],
    managed_messages: Sequence[Dict[str, str]],
    stats: Dict[str, object],
) -> None:
    """Print a detailed report for a strategy."""
    print("\n" + "=" * 90)
    print(f"📊 STRATEGY: {spec.name}")
    print(f"   Description: {spec.description}")
    print("-" * 90)
    
    # Stats
    relevant_stats = {k: v for k, v in stats.items() if k in {
        "strategy", "max_messages", "max_tokens", "reserve_tokens",
        "total_messages", "messages_in_context", "messages_dropped",
        "tokens_before", "tokens_after", "tokens_saved", "compression_ratio",
    }}
    print(f"📈 Stats: {relevant_stats}")
    
    print(f"\n📝 Original messages: {len(original_messages)}")
    print(f"📝 Messages after strategy: {len(managed_messages)}")
    print(f"📝 Messages dropped: {len(original_messages) - len(managed_messages)}")
    
    print("\n🔍 CONTEXT PREVIEW (messages kept by strategy):")
    print(format_context_preview(managed_messages))
    print("-" * 90)


async def run_experiment_for_strategy(
    spec: StrategySpec,
    xmpp_server: str,
    provider: LLMProvider,
    experiment_num: int,
) -> Dict[str, object]:
    """Run an experiment for a single context management strategy."""
    
    print(f"\n{'#' * 90}")
    print(f"# EXPERIMENT {experiment_num}: {spec.name}")
    print(f"{'#' * 90}")
    
    # Create unique JIDs for this experiment
    agent_jid = f"agent{experiment_num}@{xmpp_server}"
    agent_password = f"agent{experiment_num}_pass"
    
    sender_jid = f"sender{experiment_num}@{xmpp_server}"
    sender_password = f"sender{experiment_num}_pass"
    
    # Create LLMAgent with this strategy
    agent = LLMAgent(
        jid=agent_jid,
        password=agent_password,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        context_management=spec.context_management,
        termination_markers=["[END]"],
        verify_security=False,
    )
    
    # Track received responses
    responses_received = []
    response_event = asyncio.Event()
    
    def on_response(message: str, sender: str):
        responses_received.append(message)
        response_event.set()
    
    # Create a simple chat agent to send messages
    chat = ChatAgent(
        jid=sender_jid,
        password=sender_password,
        target_agent_jid=agent_jid,
        display_callback=on_response,
        verify_security=False,
    )
    
    try:
        # Start agents
        await agent.start()
        await asyncio.sleep(0.5)
        await chat.start()
        await asyncio.sleep(0.5)
        
        print(f"\n🚀 Agents started: {agent_jid} <-> {sender_jid}")
        print(f"📨 Sending {len(TEST_MESSAGES)} test messages...\n")
        
        # Send test messages and wait for responses
        for i, msg_text in enumerate(TEST_MESSAGES):
            print(f"  → Message {i+1}: {msg_text[:60]}...")
            response_event.clear()
            
            chat.send_message(msg_text)
            
            # Wait for response with timeout
            try:
                await asyncio.wait_for(response_event.wait(), timeout=30.0)
                print(f"  ← Response {i+1}: {responses_received[-1][:60]}...")
            except asyncio.TimeoutError:
                print(f"  ⚠ Timeout waiting for response {i+1}")
            
            await asyncio.sleep(0.3)
        
        # Get the context from the agent
        print("\n📊 Analyzing context after conversation...")
        
        # Access the context manager to see what was kept
        context_manager = agent.context
        
        # Get conversation ID (typically based on sender JID)
        conversation_id = sender_jid
        
        # Get all messages in the conversation
        if conversation_id in context_manager._conversations:
            original_messages = list(context_manager._conversations[conversation_id])
        else:
            # Try to find any conversation
            if context_manager._conversations:
                conv_id = list(context_manager._conversations.keys())[0]
                original_messages = list(context_manager._conversations[conv_id])
                conversation_id = conv_id
            else:
                original_messages = []
        
        # Apply the strategy to see what would be sent
        if original_messages:
            # Check if apply_context_strategy is async (needed for ResumeTokenContext)
            apply_strategy = context_manager.context_management.apply_context_strategy
            if asyncio.iscoroutinefunction(apply_strategy):
                managed_messages = await apply_strategy(
                    original_messages,
                    SYSTEM_PROMPT,
                )
            else:
                managed_messages = apply_strategy(
                    original_messages,
                    SYSTEM_PROMPT,
                )
            stats = context_manager.context_management.get_stats(
                total_messages=len(original_messages)
            )
        else:
            managed_messages = []
            stats = {}
        
        # Print the report
        print_strategy_report(spec, original_messages, managed_messages, stats)
        
        # Return metrics for final comparison
        return {
            "strategy": spec.name,
            "conversation_id": conversation_id,
            "total_messages": len(original_messages),
            "messages_kept": len(managed_messages),
            "messages_dropped": len(original_messages) - len(managed_messages),
            "total_chars": _sum_chars(original_messages),
            "kept_chars": _sum_chars(managed_messages),
            **{k: v for k, v in (stats or {}).items() if k in (
                "tokens_before", "tokens_after", "tokens_saved", "compression_ratio"
            )},
        }
        
    except Exception as e:
        print(f"❌ Error in experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await chat.stop()
        await agent.stop()
        await asyncio.sleep(0.5)


async def main():
    """Main function to run all context management experiments."""
    
    # Load environment variables
    load_env_vars()
    
    # Get configuration from environment
    LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss:20b")
    TOKEN_COUNT_MODEL = os.environ.get("TOKEN_COUNT_MODEL", "gpt-oss:20b")
    RESUME_MODEL = os.environ.get("RESUME_MODEL", "openai/gpt-oss:20b")
    
    print("=" * 90)
    print("🔬 CONTEXT MANAGEMENT STRATEGIES EXPERIMENT")
    print("=" * 90)
    print(f"\nThis experiment tests {len(build_strategies(RESUME_MODEL))} different context management strategies.")
    print(f"For each strategy, we send {len(TEST_MESSAGES)} messages and analyze the resulting context.\n")
    
    # XMPP server configuration
    xmpp_server = "localhost"
    print("🌐 Using SPADE built-in server (localhost:5222)")
    print("  Make sure to run 'spade run' in another terminal!\n")
    
    # Create provider
    provider = LLMProvider(model=LLM_MODEL)
    
    # Build all strategies
    strategies = build_strategies(RESUME_MODEL)
    
    # Run experiment for each strategy and collect metrics
    results: List[Dict[str, object]] = []
    for i, spec in enumerate(strategies, start=1):
        metrics = await run_experiment_for_strategy(spec, xmpp_server, provider, i)
        if metrics is not None:
            results.append(metrics)
        await asyncio.sleep(1)  # Small delay between experiments
    
    # Final summary
    print("\n" + "=" * 90)
    print("🏁 ALL EXPERIMENTS COMPLETED")
    print("=" * 90)
    print("\nSummary of strategies tested:")
    for i, spec in enumerate(strategies, start=1):
        print(f"  {i}. {spec.name}")
        print(f"     → {spec.description}")
    # Final comparison summary
    print("\n" + "=" * 90)
    print("📊 FINAL CONTEXT COMPARISON")
    print("=" * 90)
    for r in results:
        tokens_part = (
            f", tokens {r.get('tokens_after', '?')}/{r.get('tokens_before', '?')}"
            if ('tokens_before' in r and 'tokens_after' in r) else ""
        )
        print(
            f"- {r['strategy']}: kept {r['messages_kept']}/{r['total_messages']} msgs, "
            f"chars {r['kept_chars']}/{r['total_chars']}" + tokens_part
        )
    print("-" * 90)
    print("\n✅ Done!")


if __name__ == "__main__":
    import spade
    spade.run(main(), True)
