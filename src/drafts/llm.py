"""AI provider selection.

Two backends produce drafts:

* ``anthropic`` - the Claude API via the Anthropic SDK (needs ``ANTHROPIC_API_KEY``)
* ``copilot``   - the GitHub Copilot SDK (needs a Copilot subscription and
  ``COPILOT_GITHUB_TOKEN`` or a local ``copilot``/``gh`` login)

Resolution order: ``AI_PROVIDER`` env var, then ``llm.provider`` in
config.yaml. ``auto`` (the default, also used for unknown values) picks
``anthropic`` when ``ANTHROPIC_API_KEY`` is set and ``copilot`` otherwise.
"""

from __future__ import annotations

import asyncio
import os

PROVIDERS = ("anthropic", "copilot")


def resolve_provider(config: dict | None = None) -> str:
    requested = (os.environ.get("AI_PROVIDER") or (config or {}).get("provider") or "auto").strip().lower()
    if requested in PROVIDERS:
        return requested
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "copilot"


def run_pipeline_sync(
    system_prompt: str,
    user_prompt: str,
    critic_prompt: str | None = None,
    critic_input: str | None = None,
    config: dict | None = None,
) -> tuple[dict, str | None, dict]:
    """Run draft + critique on the selected provider. Sync wrapper for click commands."""
    config = config or {}
    if resolve_provider(config) == "anthropic":
        from src.drafts.anthropic_client import run_draft_pipeline
    else:
        from src.drafts.copilot_client import run_draft_pipeline
    parsed, critique, meta = asyncio.run(
        run_draft_pipeline(system_prompt, user_prompt, critic_prompt, critic_input, config)
    )
    meta.setdefault("provider", resolve_provider(config))
    return parsed, critique, meta
