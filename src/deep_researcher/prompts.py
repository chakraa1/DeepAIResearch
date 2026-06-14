"""Cached system prompt loading from YAML."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import yaml


@lru_cache(maxsize=1)
def load_system_prompts() -> dict[str, dict[str, str]]:
    """Load all system prompts once per process."""

    prompt_file = resources.files("deep_researcher").joinpath("system_prompts.yaml")
    with prompt_file.open("r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file) or {}
    return prompts


@lru_cache(maxsize=32)
def get_system_prompt(prompt_key: str) -> str:
    """Return a cached prompt with its explicit agent role."""

    prompts = load_system_prompts()
    if prompt_key not in prompts:
        raise KeyError(f"Unknown system prompt: {prompt_key}")

    entry = prompts[prompt_key]
    role = entry.get("role", prompt_key)
    prompt = entry.get("prompt", "")
    return f"Role: {role}\n\n{prompt}".strip()


def render_system_prompt(prompt_key: str, **values: object) -> str:
    """Render a cached prompt template with runtime values."""

    return get_system_prompt(prompt_key).format(**values)
