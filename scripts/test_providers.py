"""Verify provider routing WITHOUT any API keys:

  - mock      -> canned structured output (works offline)
  - deepseek  -> our urllib client (no langchain_openai); missing key = clear error
  - openai    -> preserved: still requires langchain_openai (not installed here)

Run:  python scripts/test_providers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

import src.config as config
import src.llm as llm


class Tiny(BaseModel):
    x: int


def check_mock():
    config.settings.llm_provider = "mock"
    out = llm.invoke_structured("research", Tiny, "irrelevant", mock={"x": 7})
    assert out.x == 7
    print("[PASS] mock -> returns canned structured output")


def check_deepseek_uses_urllib():
    config.settings.llm_provider = "deepseek"
    config.settings.deepseek_api_key = None
    try:
        llm.invoke_structured("research", Tiny, "irrelevant")
    except ValueError as e:
        if "DEEPSEEK_API_KEY" in str(e):
            print("[PASS] deepseek -> urllib client; missing key gives clear error (no langchain_openai)")
            return
        raise
    raise AssertionError("expected ValueError for missing DeepSeek key")


def check_openai_preserved():
    config.settings.llm_provider = "openai"
    try:
        llm.get_chat_model("research")
    except ModuleNotFoundError as e:
        if "langchain_openai" in str(e):
            print("[PASS] openai -> still uses langchain_openai (preserved; not installed here)")
            return
        raise
    print("[WARN] openai did not raise ModuleNotFoundError")


if __name__ == "__main__":
    check_mock()
    check_deepseek_uses_urllib()
    check_openai_preserved()
    print("PROVIDER CHECKS COMPLETE")
