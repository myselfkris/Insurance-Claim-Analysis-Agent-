"""Provider-agnostic chat-model factory with lazy imports + a mock provider.

Providers: openai | anthropic | google | ollama | deepseek | mock
The `mock` provider returns canned structured outputs so the pipeline runs
end-to-end BEFORE you have an API key.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings


def get_chat_model(role: str, temperature: float = 0.0) -> BaseChatModel:
    provider = settings.llm_provider
    model = settings.model_for(role)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key)

    if provider == "deepseek":
        # DeepSeek now uses our own urllib client (no langchain-openai needed).
        raise ValueError("deepseek provider has no LangChain chat model; use invoke_structured().")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, api_key=settings.anthropic_api_key)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=settings.google_api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)

    if provider == "mock":
        raise ValueError("mock provider has no chat model; use invoke_structured() instead.")

    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}. Use openai|anthropic|google|ollama|deepseek|mock.")


def structured_model(role: str, schema, temperature: float = 0.0):
    return get_chat_model(role, temperature=temperature).with_structured_output(schema)


def invoke_structured(role: str, schema, messages, mock: dict | None = None, temperature: float = 0.0):
    """Return a structured output (canned for mock, urllib for deepseek, else LangChain)."""
    provider = settings.llm_provider
    if provider == "mock":
        if mock is None:
            raise ValueError("mock provider requires a mock= dict for each call.")
        return schema(**mock)
    if provider == "deepseek":
        return _deepseek_structured(role, schema, messages, temperature)
    return structured_model(role, schema, temperature).invoke(messages)


def _deepseek_structured(role: str, schema, prompt: str, temperature: float = 0.0):
    """Call DeepSeek's OpenAI-compatible API directly with urllib (no install needed)."""
    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set. Add it to .env or set the env var.")

    model = settings.model_for(role)
    schema_desc = json.dumps(schema.model_json_schema())
    content = (
        "Return ONLY a JSON object (no markdown fences, no extra text) that matches this JSON schema:\n"
        f"{schema_desc}\n\n"
        f"Task:\n{prompt}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "stream": False,
    }
    url = settings.deepseek_base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {e.code}: {detail}") from e

    content_out = body["choices"][0]["message"]["content"]
    data = json.loads(content_out)
    return schema(**data)
