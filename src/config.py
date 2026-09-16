"""Central configuration loaded from environment / .env.

Everything provider-specific is read here so the rest of the code stays agnostic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Redirect the HuggingFace model cache into the workspace (the sandbox cannot
# write to ~/.cache). Must be set before any sentence_transformers import.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache"))

_DEFAULT_MODELS = {
    "research": "gpt-4o-mini",
    "draft": "gpt-4o",
    "verify": "gpt-4o",
}


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or None
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    embeddings_mode: str = os.getenv("EMBEDDINGS", "local").lower()
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    openai_embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

    max_cycles: int = int(os.getenv("MAX_CYCLES", "2"))
    max_research_rounds: int = int(os.getenv("MAX_RESEARCH_ROUNDS", "2"))

    db_path: Path = PROJECT_ROOT / os.getenv("DB_PATH", "data/cases.db")

    def model_for(self, role: str) -> str:
        override = os.getenv(f"MODEL_{role.upper()}")
        if override:
            return override
        if self.llm_provider == "deepseek":
            return "deepseek-chat"
        return _DEFAULT_MODELS.get(role, "gpt-4o-mini")


settings = Settings()
