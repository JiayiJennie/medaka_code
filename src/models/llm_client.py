"""LLM client implementations for different providers."""
from abc import ABC, abstractmethod
import os
import random
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import openai
from src.utils.config import Config

class LLMClient(ABC):
    """Base abstract class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """Generate a completion for the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_tokens: Maximum number of tokens to generate
            assistant_prefill: Optional text to prefill the assistant response (Anthropic only)
            
        Returns:
            Tuple of generated text and usage metadata
        """
        pass


def _empty_usage() -> Dict[str, int]:
    return {"input": 0, "output": 0, "total": 0, "reasoning": 0}


def _usage_from_openai_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return _empty_usage()
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    input_tokens = int(prompt_tokens or 0)
    output_tokens = int(completion_tokens or 0)
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        if not reasoning_tokens:
            reasoning_tokens = int(getattr(details, "thinking_tokens", 0) or 0)
    elif isinstance(usage, dict):
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or details.get("thinking_tokens", 0) or 0)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": int(total_tokens or 0),
        "reasoning": reasoning_tokens,
    }


class AzureOpenAIClient(LLMClient):
    """Client for Azure OpenAI API."""
    
    def __init__(self):
        """Initialize the Azure OpenAI client."""
        # Configure base URL and API key
        openai.api_type = "azure"
        openai.api_base = Config.AZURE_OPENAI_ENDPOINT
        openai.api_version = Config.AZURE_OPENAI_API_VERSION
        openai.api_key = Config.AZURE_OPENAI_API_KEY
        self.deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """Generate a completion using Azure OpenAI."""
        try:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a planning assistant that solves problems step by step. Follow these guidelines:\n1. Clearly prefix each action with 'ACTION:'\n2. After significant steps, show the current state with 'STATE x.y:' notation\n3. Include exploration and backtracking when appropriate\n4. End with a 'Plan summary:' section that lists all actions in order\n5. Be precise and detailed in your reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = _usage_from_openai_response(response)
            return response.choices[0].message.content, usage
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {str(e)}")
            return f"Error generating response: {str(e)}", _empty_usage()


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, reasoning_effort: Optional[str] = None):
        self.model = Config.OPENAI_MODEL_NAME
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """Generate a completion using OpenAI."""
        try:
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a planning assistant that solves problems step by step."},
                    {"role": "user", "content": prompt}
                ],
            }

            if self.reasoning_effort and self.reasoning_effort != "none":
                request["reasoning_effort"] = self.reasoning_effort
            else:
                request["temperature"] = temperature

            # Newer OpenAI models require max_completion_tokens; older ones may still use max_tokens.
            try:
                response = self.client.chat.completions.create(
                    **request,
                    max_completion_tokens=max_tokens,
                )
            except Exception as token_err:
                if "max_completion_tokens" not in str(token_err):
                    raise
                response = self.client.chat.completions.create(
                    **request,
                    max_tokens=max_tokens,
                )
            usage = _usage_from_openai_response(response)
            return response.choices[0].message.content, usage
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return f"Error generating response: {str(e)}", _empty_usage()


class OpenRouterClient(LLMClient):
    """Client for OpenRouter (OpenAI-compatible Chat Completions API)."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

    def __init__(self, enable_thinking: Optional[bool] = None):
        self.model = Config.OPENROUTER_MODEL_NAME
        self._enable_thinking = enable_thinking
        headers = {}
        if Config.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = Config.OPENROUTER_HTTP_REFERER
        if Config.OPENROUTER_APP_TITLE:
            headers["X-Title"] = Config.OPENROUTER_APP_TITLE
        client_kwargs: Dict[str, Any] = {
            "api_key": Config.OPENROUTER_API_KEY,
            "base_url": Config.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1",
        }
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = openai.OpenAI(**client_kwargs)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        user_content = prompt
        if self._enable_thinking is False and "qwen" in self.model.lower():
            user_content = "/no_think\n\n" + prompt
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a planning assistant that solves problems step by step."},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**request)
                usage = _usage_from_openai_response(response)
                return response.choices[0].message.content, usage
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    print(f"OpenRouter API attempt {attempt + 1} failed ({e}), retrying in {wait:.0f}s...")
                    time.sleep(wait)
        print(f"Error calling OpenRouter API after {self.MAX_RETRIES} attempts: {last_err}")
        return f"Error generating response: {last_err}", _empty_usage()


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API.

    When ``enable_thinking`` is True, uses Claude extended thinking. The API
    requires ``temperature=1``, ``thinking.budget_tokens >= 1024``, and that
    ``budget_tokens < max_tokens``. Thinking text (for non-redacted blocks) is
    prepended to the visible answer between ``[THINKING]`` markers so it is
    preserved in the saved JSON; the planner only reads from the last
    ``Plan summary:`` so this does not interfere with parsing.
    """

    def __init__(self, enable_thinking: Optional[bool] = None,
                 thinking_budget_tokens: int = 10000):
        import anthropic
        self.model = Config.ANTHROPIC_MODEL_NAME
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget_tokens = int(thinking_budget_tokens)

    def _count_tokens(self, text: str) -> Optional[int]:
        """Use Anthropic count_tokens API to get exact BPE token count for `text`.

        Returns None if the call fails (network/SDK incompat). The endpoint is
        free and uses the same tokenizer as a real generation, so the result
        equals the output_tokens that would be charged for the same text
        (modulo a tiny per-message envelope overhead, ~3-5 tokens).
        """
        if not text:
            return 0
        msg = [{"role": "user", "content": text}]
        try:
            resp = self.client.messages.count_tokens(model=self.model, messages=msg)
        except Exception:
            try:
                resp = self.client.beta.messages.count_tokens(model=self.model, messages=msg)
            except Exception:
                return None
        try:
            return int(getattr(resp, "input_tokens", 0) or 0)
        except Exception:
            return None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        try:
            messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
            if assistant_prefill and not self.enable_thinking:
                messages.append({"role": "assistant", "content": assistant_prefill})

            kwargs: Dict[str, Any] = dict(
                model=self.model,
                max_tokens=max_tokens,
                system="You are a planning assistant. Follow the user-provided format exactly.",
                messages=messages,
            )
            if self.enable_thinking:
                budget = max(1024, min(self.thinking_budget_tokens, max(max_tokens - 2048, 1024)))
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                kwargs["temperature"] = 1.0  # API requires T=1 with thinking
            else:
                kwargs["temperature"] = temperature

            with self.client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()

            text_parts: List[str] = []
            thinking_parts: List[str] = []
            redacted_count = 0
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif btype == "thinking":
                    thinking_parts.append(getattr(block, "thinking", "") or "")
                elif btype == "redacted_thinking":
                    redacted_count += 1
            text = "".join(text_parts)
            if assistant_prefill and not self.enable_thinking:
                text = assistant_prefill + text

            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

            # Anthropic does NOT return a separate thinking-token counter in
            # the response. Use the free count_tokens endpoint (same Claude
            # tokenizer) to get the exact BPE token count of the thinking
            # text. If that call fails for some reason, leave reasoning=0.
            thinking_text = "\n".join(t for t in thinking_parts if t)
            reasoning_tokens = 0
            if thinking_text:
                exact = self._count_tokens(thinking_text)
                if exact is not None:
                    reasoning_tokens = exact

            if thinking_text or redacted_count:
                header_lines = ["[THINKING]"]
                if thinking_text:
                    header_lines.append(thinking_text)
                else:
                    header_lines.append("(redacted thinking)")
                if redacted_count and thinking_text:
                    header_lines.append(f"(plus {redacted_count} redacted_thinking block(s))")
                header_lines.append(
                    f"(reasoning_tokens={reasoning_tokens}, output_tokens={output_tokens})"
                )
                header_lines.append("[/THINKING]")
                text = "\n".join(header_lines) + "\n\n" + text

            return text, {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
                "reasoning": reasoning_tokens,
            }
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            return f"Error generating response: {str(e)}", _empty_usage()



class GoogleClient(LLMClient):
    """Client for Google Gemini API."""

    def __init__(self):
        from google import genai
        self.model_name = Config.GOOGLE_MODEL_NAME
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        from google.genai import types
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a planning assistant that solves problems step by step.",
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                ),
            )
            usage_metadata = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(usage_metadata, "total_token_count", 0) or 0)
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens
            thinking_tokens = int(getattr(usage_metadata, "thinking_token_count", 0) or 0)
            return response.text, {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
                "reasoning": thinking_tokens,
            }
        except Exception as e:
            print(f"Error calling Google Gemini API: {str(e)}")
            return f"Error generating response: {str(e)}", _empty_usage()


class DashScopeClient(LLMClient):
    """Client for Alibaba DashScope API (Qwen models). OpenAI-compatible."""

    _BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    _THINKING_MODELS = ("deepseek-r1", "qwq")

    def __init__(self, enable_thinking: Optional[bool] = None):
        self.model = Config.DASHSCOPE_MODEL_NAME
        self.client = openai.OpenAI(
            api_key=Config.DASHSCOPE_API_KEY,
            base_url=self._BASE_URL,
        )
        if enable_thinking is not None:
            self._thinking = enable_thinking
        else:
            self._thinking = any(t in self.model.lower() for t in self._THINKING_MODELS)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        try:
            messages = [{"role": "user", "content": prompt}]
            if not self._thinking:
                messages.insert(0, {"role": "system", "content": "You are a planning assistant that solves problems step by step."})

            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": {"enable_thinking": self._thinking},
            }
            if not self._thinking:
                request_kwargs["temperature"] = temperature

            chunks = []
            reasoning_chunks = []
            usage_info = _empty_usage()
            for chunk in self.client.chat.completions.create(**request_kwargs):
                if chunk.usage:
                    if os.environ.get("DEBUG_USAGE"):
                        print(f"  [DashScope usage] {chunk.usage}")
                    usage_info = _usage_from_openai_response(chunk)
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        chunks.append(delta.content)
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_chunks.append(rc)

            if usage_info["reasoning"] == 0 and reasoning_chunks:
                reasoning_text = "".join(reasoning_chunks)
                usage_info["reasoning"] = len(reasoning_text) // 4

            result = "".join(chunks)
            if not result and not reasoning_chunks:
                print("  [DashScope] WARNING: API returned empty response (possible rate limit)")
            return result, usage_info
        except Exception as e:
            print(f"  [DashScope] ERROR: {e}")
            return f"Error generating response: {str(e)}", _empty_usage()


class TogetherClient(LLMClient):
    """Client for Together AI (OpenAI-compatible Chat Completions API)."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

    def __init__(self):
        self.model = Config.TOGETHER_MODEL_NAME
        self.client = openai.OpenAI(
            api_key=Config.TOGETHER_API_KEY,
            base_url=Config.TOGETHER_BASE_URL or "https://api.together.xyz/v1",
        )

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a planning assistant that solves problems step by step."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**request)
                usage = _usage_from_openai_response(response)
                return response.choices[0].message.content, usage
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    print(f"Together API attempt {attempt + 1} failed ({e}), retrying in {wait:.0f}s...")
                    time.sleep(wait)
        print(f"Error calling Together API after {self.MAX_RETRIES} attempts: {last_err}")
        return f"Error generating response: {last_err}", _empty_usage()


class FeatherlessClient(LLMClient):
    """Client for Featherless AI (OpenAI-compatible Chat Completions API)."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

    def __init__(self, enable_thinking: Optional[bool] = None):
        self.model = Config.FEATHERLESS_MODEL_NAME
        self._enable_thinking = enable_thinking if enable_thinking is not None else False
        self.client = openai.OpenAI(
            api_key=Config.FEATHERLESS_API_KEY,
            base_url=Config.FEATHERLESS_BASE_URL or "https://api.featherless.ai/v1",
            timeout=300.0,
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 16000,
                 assistant_prefill: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        user_content = prompt
        if not self._enable_thinking and "qwen" in self.model.lower():
            user_content = "/no_think\n\n" + prompt

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a planning assistant that solves problems step by step."},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**request)
                usage = _usage_from_openai_response(response)
                return response.choices[0].message.content, usage
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    print(f"Featherless API attempt {attempt + 1} failed ({e}), retrying in {wait:.0f}s...")
                    time.sleep(wait)
        print(f"Error calling Featherless API after {self.MAX_RETRIES} attempts: {last_err}")
        return f"Error generating response: {last_err}", _empty_usage()


def get_llm_client(provider: Optional[str] = None, reasoning_effort: Optional[str] = None,
                   enable_thinking: Optional[bool] = None) -> LLMClient:
    """Factory function to get the appropriate LLM client based on available configurations."""
    if provider is None:
        provider = Config.get_default_provider()

    if provider is None:
        raise ValueError("No LLM provider is configured. Please check your .env file.")

    if provider == "azure_openai":
        return AzureOpenAIClient()
    elif provider == "openai":
        return OpenAIClient(reasoning_effort=reasoning_effort)
    elif provider == "anthropic":
        return AnthropicClient(enable_thinking=enable_thinking)
    elif provider == "google":
        return GoogleClient()
    elif provider == "dashscope":
        return DashScopeClient(enable_thinking=enable_thinking)
    elif provider == "openrouter":
        return OpenRouterClient(enable_thinking=enable_thinking)
    elif provider == "together":
        return TogetherClient()
    elif provider == "featherless":
        return FeatherlessClient(enable_thinking=enable_thinking)
    else:
        raise ValueError(f"Unknown provider: {provider}") 