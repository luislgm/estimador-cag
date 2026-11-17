"""Plantillas de prompt versionadas. Ver app/prompts/loader.py."""

from app.prompts.loader import (
    PROMPT_VERSION,
    render_critic_prompt,
    render_estimation_prompt,
    render_examples,
    render_summary_prompt,
)

__all__ = [
    "PROMPT_VERSION",
    "render_critic_prompt",
    "render_estimation_prompt",
    "render_examples",
    "render_summary_prompt",
]
