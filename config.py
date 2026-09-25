"""
Shared settings loaded once from environment / .env file.
All other modules import from here instead of touching os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FIXION_",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(alias="GEMINI_API_KEY", default="")
    model: str = "gemini-2.5-pro"
    embed_model: str = "text-embedding-004"

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str = Field(alias="GITHUB_TOKEN", default="")

    # ── Agent loop ────────────────────────────────────────────────────────────
    max_retries: int = 5

    # ── Sandbox ───────────────────────────────────────────────────────────────
    sandbox_image: str = "fixion-sandbox:latest"
    sandbox_timeout: int = 300  # seconds
    sandbox_memory: str = "2g"

    # ── Storage ───────────────────────────────────────────────────────────────
    index_dir: Path = Path(".fixion_cache/indexes")
    workspace_dir: Path = Path(".fixion_cache/workspaces")


# Singleton — import this everywhere
settings = Settings()
