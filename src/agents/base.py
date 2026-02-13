import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from loguru import logger

from src.models import AgentRole, AgentResponse
from src.utils.exceptions import LLMError


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(f"OllamaClient initialized: {base_url} with model {model}")

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        try:
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            if system:
                payload["system"] = system

            start_time = time.time()
            logger.debug(f"Generating with Ollama: {len(prompt)} chars prompt")

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            elapsed = time.time() - start_time

            response_text = data.get("response", "")
            total_duration = data.get("total_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)

            logger.info(
                f"Ollama generation complete: {eval_count} tokens in {elapsed:.2f}s "
                f"({eval_count/elapsed:.1f} tok/s)"
            )

            return response_text

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            raise LLMError(
                f"Ollama request failed with status {e.response.status_code}",
                model=self.model,
            )
        except httpx.RequestError as e:
            logger.error(f"Ollama connection error: {e}")
            raise LLMError(
                f"Failed to connect to Ollama at {self.base_url}",
                model=self.model,
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise LLMError(str(e), model=self.model)

    async def is_available(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            model_names = [m.get("name", "") for m in models]

            if self.model in model_names:
                logger.debug(f"Model {self.model} is available")
                return True

            for model_name in model_names:
                if self.model in model_name or model_name.startswith(self.model):
                    logger.debug(f"Model {self.model} found as {model_name}")
                    return True

            logger.warning(f"Model {self.model} not found in available models: {model_names}")
            return False

        except Exception as e:
            logger.warning(f"Failed to check Ollama availability: {e}")
            return False

    async def list_models(self) -> list[str]:
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            model_names = [m.get("name", "") for m in models]
            logger.info(f"Available models: {model_names}")

            return model_names

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def can_generate(self) -> bool:
        """
        Run a minimal generation request as a health check.

        This catches cases where `/api/tags` works but `/api/generate` fails.
        """
        try:
            await self.generate(
                prompt="Reply with exactly: OK",
                temperature=0.0,
                max_tokens=8,
            )
            logger.debug(f"Model {self.model} passed generation health check")
            return True
        except LLMError as e:
            logger.warning(f"Model {self.model} failed generation health check: {e}")
            return False
        except Exception as e:
            logger.warning(
                f"Unexpected generation health check failure for model {self.model}: {e}"
            )
            return False

    async def close(self) -> None:
        await self.client.aclose()
        logger.info("OllamaClient closed")

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


def load_prompt(agent_name: str) -> str:
    try:
        prompts_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        prompt_file = prompts_dir / f"{agent_name}.yaml"

        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, "r") as f:
            data = yaml.safe_load(f)

        system_prompt = data.get("system", "")

        if not system_prompt:
            logger.warning(f"Empty system prompt in {prompt_file}")

        logger.debug(f"Loaded prompt for {agent_name}: {len(system_prompt)} chars")
        return system_prompt

    except Exception as e:
        logger.error(f"Failed to load prompt for {agent_name}: {e}")
        raise


class BaseAgent:
    def __init__(
        self,
        role: AgentRole,
        ollama: OllamaClient,
        system_prompt: str,
    ) -> None:
        self.role = role
        self.ollama = ollama
        self.system_prompt = system_prompt

        logger.info(f"BaseAgent initialized for role: {role}")

    async def generate(
        self,
        context: str,
        instruction: str,
        **kwargs: Any,
    ) -> AgentResponse:
        start_time = time.time()

        prompt = self._build_prompt(context, instruction, **kwargs)

        logger.debug(f"{self.role} generating response")

        response_text = await self.ollama.generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
        )

        elapsed = time.time() - start_time

        agent_response = AgentResponse(
            role=self.role,
            round_number=kwargs.get("round_number", 1),
            content=response_text,
            probability=self._extract_probability(response_text),
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"{self.role} response generated in {elapsed:.2f}s")

        return agent_response

    def _build_prompt(
        self,
        context: str,
        instruction: str,
        **kwargs: Any,
    ) -> str:
        parts = []

        parts.append("# Context")
        parts.append(context)
        parts.append("")

        if "previous_arguments" in kwargs and kwargs["previous_arguments"]:
            parts.append("# Previous Arguments")
            for arg in kwargs["previous_arguments"]:
                parts.append(f"## {arg['role']}")
                parts.append(arg["content"])
                parts.append("")

        parts.append("# Your Task")
        parts.append(instruction)

        return "\n".join(parts)

    def _extract_probability(self, text: str) -> float | None:
        try:
            lines = text.split("\n")
            for line in lines:
                line_lower = line.lower()
                if "p(yes)" in line_lower or "probability" in line_lower:
                    for word in line.split():
                        word_clean = word.strip(":%,")
                        if "." in word_clean or word_clean.isdigit():
                            try:
                                prob = float(word_clean)
                                if 0 <= prob <= 1:
                                    return prob
                                if 0 <= prob <= 100:
                                    return prob / 100
                            except ValueError:
                                continue

            return None

        except Exception as e:
            logger.warning(f"Failed to extract probability: {e}")
            return None
