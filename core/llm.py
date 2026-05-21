import litellm
from settings import settings
from logger import setup_logger

logger = setup_logger(__name__)

litellm.set_verbose = False


def _build_model_string() -> str:
    """
    Constructs the LiteLLM model string from provider and model settings.

    LiteLLM requires provider-prefixed strings:
        huggingface/deepseek-ai/DeepSeek-V3-0324
        ollama/qwen2.5-coder:7b-instruct
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        return f"ollama/{settings.OLLAMA_MODEL}"
    elif provider == "huggingface":
        return f"huggingface/{settings.LLM_MODEL}"
    else:
        # Passthrough for openai, anthropic, etc.
        return settings.LLM_MODEL


class LLMService:
    """
    LiteLLM-backed LLM abstraction.
    Supports huggingface, ollama, and any LiteLLM-compatible provider.
    """

    def __init__(self) -> None:
        self._model = _build_model_string()
        self._temperature = settings.TEMPERATURE
        self._max_tokens = settings.MAX_TOKENS

        if settings.LLM_PROVIDER.lower() == "ollama":
            litellm.api_base = settings.OLLAMA_BASE_URL

    def complete(self, prompt: str) -> str:
        logger.info(f"Sending prompt to {self._model} ({len(prompt)} chars)")
        try:
            response = litellm.completion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                api_key=settings.HF_TOKEN if settings.LLM_PROVIDER == "huggingface" else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            return "⚠️ Failed to generate a response from the LLM."