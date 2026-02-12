#!/usr/bin/env python3
"""
Test script for Ollama agents integration.
"""
import asyncio
from config.settings import get_settings
from src.agents import OllamaClient, BaseAgent, load_prompt
from src.models import AgentRole


async def test_ollama_connection():
    print("=" * 80)
    print("Testing Ollama Connection")
    print("=" * 80)

    settings = get_settings()

    async with OllamaClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=120
    ) as ollama:
        # Check if Ollama is available
        is_available = await ollama.is_available()
        print(f"\nOllama Status: {'AVAILABLE' if is_available else 'OFFLINE'}")

        if not is_available:
            print("\nOllama is not available. Please ensure:")
            print("1. Ollama is running (ollama serve)")
            print(f"2. Model '{settings.llm.model}' is pulled (ollama pull {settings.llm.model})")
            return

        # List available models
        models = await ollama.list_models()
        print(f"\nAvailable Models: {', '.join(models)}")


async def test_prompt_loading():
    print("\n" + "=" * 80)
    print("Testing Prompt Loading")
    print("=" * 80)

    agents = ["bull_agent", "bear_agent", "devil_agent", "judge_agent"]

    for agent_name in agents:
        try:
            prompt = load_prompt(agent_name)
            print(f"\n{agent_name}: {len(prompt)} characters")
            print(f"  First 100 chars: {prompt[:100]}...")
        except Exception as e:
            print(f"\n{agent_name}: ERROR - {e}")


async def test_agent_generation():
    print("\n" + "=" * 80)
    print("Testing Agent Generation")
    print("=" * 80)

    settings = get_settings()

    async with OllamaClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=120
    ) as ollama:
        is_available = await ollama.is_available()

        if not is_available:
            print("\nSkipping generation test - Ollama not available")
            return

        print("\nGenerating response with Bull Agent...")

        system_prompt = load_prompt("bull_agent")
        agent = BaseAgent(
            role=AgentRole.BULL,
            ollama=ollama,
            system_prompt=system_prompt,
        )

        context = """
# Market Analysis Context

## Question
Will SpaceX successfully land humans on Mars before 2030?

## Current Market State
- Current Price (Market P(YES)): 15.0%
- Volume 24h: $50,000
- Liquidity: $200,000
- Days Remaining: 1825

## Recent News
- [Space News] SpaceX completes successful Starship test (2026-02-10) - Sentiment: Positive
- [Tech Times] Mars mission faces budget challenges (2026-02-08) - Sentiment: Negative
"""

        instruction = "Provide your opening argument for why this market will resolve YES."

        response = await agent.generate(
            context=context,
            instruction=instruction,
            round_number=1,
            temperature=0.7,
            max_tokens=500,
        )

        print(f"\nAgent: {response.role}")
        print(f"Round: {response.round_number}")
        print(f"Probability: {response.probability}")
        print(f"\nResponse (first 500 chars):")
        print(response.content[:500])
        print("...")


async def main():
    try:
        await test_ollama_connection()
        await test_prompt_loading()
        await test_agent_generation()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
