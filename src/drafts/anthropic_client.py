"""Claude API (Anthropic SDK) backend for draft generation and research.

Alternative to the GitHub Copilot SDK backend in ``copilot_client.py``.
Selected when ``ANTHROPIC_API_KEY`` is set (or ``AI_PROVIDER=anthropic`` /
``llm.provider: anthropic`` in config.yaml). Exposes the same
``run_draft_pipeline`` contract as the Copilot backend so the drafter does
not need to know which provider is in use.
"""

from __future__ import annotations

import json
import logging

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_DRAFT_MODEL = "claude-opus-5-5"
DEFAULT_CRITIC_MODEL = "claude-opus-5-5"

# Server-side refusal fallback: if the primary model declines, the API
# re-runs the request on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_TOKENS = 16000
RESEARCH_MAX_TOOL_CALLS = 5


def _is_anthropic_model(name: str | None) -> bool:
    return bool(name) and str(name).startswith("claude-") and "." not in str(name)


def _pick_model(config: dict, key: str, default: str) -> str:
    """Use a configured model only if it is a Claude API model id.

    Copilot-style ids such as ``claude-opus-4.6`` or ``gpt-5.4`` are ignored
    so that switching providers never sends an invalid model id.
    """
    value = config.get(key)
    return value if _is_anthropic_model(value) else default


def _text_of(message) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


async def _generate(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_prompt: str,
    model: str,
    effort: str,
) -> tuple[str, str]:
    """Single streamed request. Returns (text, model_that_served_it)."""
    async with client.beta.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"effort": effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        message = await stream.get_final_message()

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise RuntimeError(f"Model declined the request (category={category})")
    if message.stop_reason == "max_tokens":
        logger.warning("Response hit max_tokens; output may be truncated")

    text = _text_of(message)
    if not text.strip():
        raise RuntimeError(f"Empty response from {model}")
    return text, message.model


async def run_draft_pipeline(
    system_prompt: str,
    user_prompt: str,
    critic_prompt: str | None,
    critic_input: str | None,
    config: dict,
) -> tuple[dict, str | None, dict]:
    """Draft + critique via the Claude API.

    Returns (parsed_draft_dict, critique_text_or_none, metadata), matching
    ``copilot_client.run_draft_pipeline``.
    """
    from src.drafts.drafter import _parse_llm_json

    draft_model = _pick_model(config, "draft_model", DEFAULT_DRAFT_MODEL)
    critic_model = _pick_model(config, "critic_model", DEFAULT_CRITIC_MODEL)

    async with anthropic.AsyncAnthropic(max_retries=4) as client:
        # Step 1: draft. Retry once if the model returns unparseable JSON.
        draft_parsed = None
        draft_model_used = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw, draft_model_used = await _generate(
                    client, system_prompt, user_prompt, draft_model, effort="high"
                )
                draft_parsed = _parse_llm_json(raw)
                break
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning("Draft JSON parse failed (attempt %d): %s", attempt + 1, str(e)[:100])
            except anthropic.AuthenticationError as e:
                raise RuntimeError("Claude API auth failed: check ANTHROPIC_API_KEY") from e
            except anthropic.APIError as e:
                last_error = e
                logger.warning("Claude API error (attempt %d): %s", attempt + 1, str(e)[:200])
        if draft_parsed is None:
            raise RuntimeError(f"All models failed. Last: {last_error}")
        logger.info("Drafted with %s", draft_model_used)

        # Step 2: critique (best-effort).
        critique_text = None
        critic_model_used = None
        if critic_prompt and critic_input:
            try:
                full_input = f"{critic_input}\n\n{draft_parsed.get('body', '')}"
                critique_text, critic_model_used = await _generate(
                    client, critic_prompt, full_input, critic_model, effort="medium"
                )
            except Exception:
                logger.warning("Critique failed, using draft as-is", exc_info=True)

    return draft_parsed, critique_text, {
        "draft_model": draft_model_used,
        "critic_model": critic_model_used,
    }


# ---------------------------------------------------------------------------
# Research agent (tool use)
# ---------------------------------------------------------------------------

RESEARCH_TOOLS = [
    {
        "name": "fetch_article",
        "description": "Fetch the readable text of an HTTPS article for fact verification.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "HTTPS URL to fetch"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_learn",
        "description": "Search Microsoft Learn documentation and return the top results.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_terraform",
        "description": "Verify that a Terraform resource type exists in a provider's registry docs.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "description": "Provider name, e.g. azurerm"},
                "resource_type": {"type": "string", "description": "Resource type, e.g. kubernetes_cluster"},
            },
            "required": ["provider", "resource_type"],
            "additionalProperties": False,
        },
    },
]


def _run_research_tool(name: str, args: dict) -> str:
    from src.feeds.research_tools import (
        check_terraform_resource,
        fetch_article,
        search_microsoft_learn,
    )

    if name == "fetch_article":
        return fetch_article(args["url"])
    if name == "search_learn":
        return search_microsoft_learn(args["query"])
    if name == "check_terraform":
        return check_terraform_resource(args["provider"], args["resource_type"])
    raise ValueError(f"Unknown tool: {name}")


async def run_research_session(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
    """Manual tool-use loop for the research agent. Returns the final text."""
    model = model if _is_anthropic_model(model) else DEFAULT_DRAFT_MODEL
    messages: list = [{"role": "user", "content": user_prompt}]
    tool_calls = 0

    async with anthropic.AsyncAnthropic(max_retries=3) as client:
        for _ in range(RESEARCH_MAX_TOOL_CALLS + 3):
            response = await client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=RESEARCH_TOOLS,
                messages=messages,
                output_config={"effort": "low"},
            )
            if response.stop_reason != "tool_use":
                return _text_of(response)

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                if tool_calls > RESEARCH_MAX_TOOL_CALLS:
                    content, is_error = "Tool budget exhausted. Return your JSON summary now.", True
                else:
                    try:
                        content, is_error = _run_research_tool(block.name, block.input), False
                    except Exception as e:  # tool failures go back to the model
                        content, is_error = f"Tool error: {e}", True
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content[:20000],
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})
    logger.warning("Research loop hit its turn limit without a final answer")
    return ""
