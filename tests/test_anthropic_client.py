"""Tests for provider selection and the Claude API backend."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.drafts import anthropic_client, llm


class TestResolveProvider:
    def test_auto_prefers_anthropic_when_key_set(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert llm.resolve_provider({"provider": "auto"}) == "anthropic"

    def test_auto_falls_back_to_copilot(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert llm.resolve_provider({}) == "copilot"

    def test_legacy_provider_value_treated_as_auto(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert llm.resolve_provider({"provider": "azure_openai"}) == "anthropic"

    def test_config_explicit_copilot_wins_over_key(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert llm.resolve_provider({"provider": "copilot"}) == "copilot"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "Anthropic")
        assert llm.resolve_provider({"provider": "copilot"}) == "anthropic"


class TestModelPicking:
    def test_copilot_style_ids_are_ignored(self):
        assert anthropic_client._pick_model({"draft_model": "claude-opus-4.6"}, "draft_model", "d") == "d"
        assert anthropic_client._pick_model({"draft_model": "gpt-5.4"}, "draft_model", "d") == "d"
        assert anthropic_client._pick_model({"draft_model": None}, "draft_model", "d") == "d"

    def test_claude_api_ids_are_used(self):
        cfg = {"draft_model": "claude-sonnet-5-5"}
        assert anthropic_client._pick_model(cfg, "draft_model", "d") == "claude-sonnet-5-5"


def _message(text, stop_reason="end_turn", model="claude-opus-5-5", content=None):
    return SimpleNamespace(
        content=content if content is not None else [SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        model=model,
    )


class _FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


class _FakeClient:
    """Stand-in for anthropic.AsyncAnthropic."""

    def __init__(self, stream_messages=None, create_messages=None):
        self.stream_messages = list(stream_messages or [])
        self.create_messages = list(create_messages or [])
        self.stream_calls = []
        self.create_calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.messages = SimpleNamespace(create=self._create)

    def _stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _FakeStream(self.stream_messages.pop(0))

    async def _create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.create_messages.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


DRAFT_JSON = json.dumps({"body": "A post body", "hashtags": ["#Azure"], "pattern_used": "share"})


class TestDraftPipeline:
    def test_draft_and_critique(self):
        fake = _FakeClient(stream_messages=[_message(DRAFT_JSON), _message("Critique text")])
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake):
            parsed, critique, meta = asyncio.run(
                anthropic_client.run_draft_pipeline("sys", "user", "critic sys", "Review:", {})
            )
        assert parsed["body"] == "A post body"
        assert critique == "Critique text"
        assert meta["draft_model"] == "claude-opus-5-5"
        first = fake.stream_calls[0]
        assert first["model"] == "claude-opus-5-5"
        assert first["fallbacks"] == "default"
        assert anthropic_client.FALLBACK_BETA in first["betas"]
        assert "temperature" not in first
        assert "A post body" in fake.stream_calls[1]["messages"][0]["content"]

    def test_retries_on_bad_json(self):
        fake = _FakeClient(stream_messages=[_message("not json at all"), _message(DRAFT_JSON)])
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake):
            parsed, critique, _ = asyncio.run(
                anthropic_client.run_draft_pipeline("sys", "user", None, None, {})
            )
        assert parsed["body"] == "A post body"
        assert critique is None
        assert len(fake.stream_calls) == 2

    def test_refusal_raises(self):
        fake = _FakeClient(stream_messages=[_message("", stop_reason="refusal")] * 2)
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake):
            with pytest.raises(RuntimeError):
                asyncio.run(anthropic_client.run_draft_pipeline("sys", "user", None, None, {}))

    def test_critique_failure_is_non_fatal(self):
        fake = _FakeClient(stream_messages=[_message(DRAFT_JSON), _message("", stop_reason="refusal")])
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake):
            parsed, critique, meta = asyncio.run(
                anthropic_client.run_draft_pipeline("sys", "user", "critic", "Review:", {})
            )
        assert parsed["body"] == "A post body"
        assert critique is None
        assert meta["critic_model"] is None


class TestResearchSession:
    def test_tool_loop(self):
        tool_use = SimpleNamespace(type="tool_use", id="t1", name="fetch_article",
                                   input={"url": "https://example.com/a"})
        fake = _FakeClient(create_messages=[
            _message("", stop_reason="tool_use", content=[tool_use]),
            _message('{"article_summary": "ok"}'),
        ])
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake), \
             patch("src.feeds.research_tools.fetch_article", return_value="ARTICLE TEXT") as fetch:
            out = asyncio.run(anthropic_client.run_research_session("sys", "user"))
        assert out == '{"article_summary": "ok"}'
        fetch.assert_called_once_with("https://example.com/a")
        tool_result = fake.create_calls[1]["messages"][-1]["content"][0]
        assert tool_result["tool_use_id"] == "t1"
        assert tool_result["content"] == "ARTICLE TEXT"
        assert tool_result["is_error"] is False

    def test_tool_errors_are_returned_to_model(self):
        tool_use = SimpleNamespace(type="tool_use", id="t1", name="nope", input={})
        fake = _FakeClient(create_messages=[
            _message("", stop_reason="tool_use", content=[tool_use]),
            _message("done"),
        ])
        with patch.object(anthropic_client.anthropic, "AsyncAnthropic", return_value=fake):
            assert asyncio.run(anthropic_client.run_research_session("sys", "user")) == "done"
        assert fake.create_calls[1]["messages"][-1]["content"][0]["is_error"] is True


class TestDispatch:
    def test_routes_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")

        async def fake_pipeline(*args):
            return {"body": "x"}, None, {"draft_model": "m"}

        with patch("src.drafts.anthropic_client.run_draft_pipeline", side_effect=fake_pipeline) as p:
            parsed, _, meta = llm.run_pipeline_sync("s", "u")
        assert p.called
        assert parsed == {"body": "x"}
        assert meta["provider"] == "anthropic"

    def test_routes_to_copilot(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "copilot")
        mock = MagicMock()

        async def fake_pipeline(*args):
            mock(*args)
            return {"body": "y"}, None, {}

        with patch("src.drafts.copilot_client.run_draft_pipeline", side_effect=fake_pipeline):
            parsed, _, meta = llm.run_pipeline_sync("s", "u")
        assert mock.called
        assert meta["provider"] == "copilot"
