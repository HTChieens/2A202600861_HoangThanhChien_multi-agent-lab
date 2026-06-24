"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client.

    The lab currently ships with an OpenAI implementation because the settings already
    expose `OPENAI_API_KEY` and `OPENAI_MODEL`. Agents should still depend only on this
    class so the provider can be swapped later without changing agent code.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self.timeout_seconds = timeout_seconds or self.settings.timeout_seconds
        self.max_retries = max_retries

        if self.max_retries < 1:
            raise ValidationError("max_retries must be at least 1")

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Retry, timeout, and token accounting live here rather than inside agents.
        """

        self._validate_prompt(system_prompt, "system_prompt")
        self._validate_prompt(user_prompt, "user_prompt")

        completion = self._build_retrying_completion()
        return completion(system_prompt, user_prompt)

    def _build_retrying_completion(self) -> Callable[[str, str], LLMResponse]:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(AgentExecutionError),
            reraise=True,
        )
        def _completion(system_prompt: str, user_prompt: str) -> LLMResponse:
            return self._complete_with_openai(system_prompt, user_prompt)

        return _completion

    def _complete_with_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.settings.openai_api_key:
            raise ValidationError(
                "OPENAI_API_KEY is not configured. Add it to your environment or `.env` file."
            )

        client = self._create_openai_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - depends on provider/network behavior.
            raise AgentExecutionError(f"LLM provider call failed: {exc}") from exc

        content = self._extract_content(response)
        input_tokens = self._extract_usage_value(response, "prompt_tokens")
        output_tokens = self._extract_usage_value(response, "completion_tokens")
        cost_usd = self._estimate_cost_usd(input_tokens, output_tokens)

        logger.info(
            "llm_completion model=%s input_tokens=%s output_tokens=%s cost_usd=%s",
            self.model,
            input_tokens,
            output_tokens,
            cost_usd,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def _create_openai_client(self) -> Any:
        try:
            openai_module = import_module("openai")
        except ImportError as exc:
            raise AgentExecutionError(
                "OpenAI SDK is not installed. Install the LLM extras with "
                '`pip install -e ".[llm]"`.'
            ) from exc

        openai_client_class = getattr(openai_module, "OpenAI", None)
        if openai_client_class is None:
            raise AgentExecutionError("Installed OpenAI SDK does not expose OpenAI client.")

        return openai_client_class(
            api_key=self.settings.openai_api_key,
            timeout=self.timeout_seconds,
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise AgentExecutionError("LLM provider returned no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise AgentExecutionError("LLM provider returned an empty completion.")

        return content

    @staticmethod
    def _extract_usage_value(response: Any, field_name: str) -> int | None:
        usage = getattr(response, "usage", None)
        value = getattr(usage, field_name, None)
        if value is None:
            return None
        if isinstance(value, int):
            return value
        return cast(int, value)

    def _estimate_cost_usd(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        input_rate = self.settings.openai_input_cost_per_1m_tokens
        output_rate = self.settings.openai_output_cost_per_1m_tokens
        if input_tokens is None or output_tokens is None:
            return None
        if input_rate is None or output_rate is None:
            return None

        return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)

    @staticmethod
    def _validate_prompt(prompt: str, field_name: str) -> None:
        if not prompt.strip():
            raise ValidationError(f"{field_name} must not be empty")
