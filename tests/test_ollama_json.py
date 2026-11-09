"""Tests for ollama_json.py. A fake replaces the network, so Ollama is not needed."""
import json

import pytest
import requests

import ollama_json as oj

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._body


def reply(content, **extra):
    return FakeResponse(body={"message": {"content": content},
                              "prompt_eval_count": 7, "eval_count": 3, **extra})


def test_strip_thinking_removes_the_reasoning_block():
    assert oj.strip_thinking('<think>let me see\nhmm</think>\n{"a": 1}') == '{"a": 1}'
    assert oj.strip_thinking('{"a": 1}') == '{"a": 1}'


def test_chat_json_returns_the_parsed_answer_and_stats(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: reply('{"a": 1}'))
    answer, stats = oj.chat_json("m", "sys", "usr", SCHEMA)
    assert answer == {"a": 1}
    assert stats["prompt_tokens"] == 7 and stats["output_tokens"] == 3


def test_chat_json_sends_the_settings_we_rely_on(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, payload=json)
        return reply('{"a": 1}')

    monkeypatch.setattr(requests, "post", fake_post)
    oj.chat_json("qwen3:latest", "SYS", "USR", SCHEMA, url="http://x:1", num_ctx=4096,
                 think=True, num_predict=99)
    p = seen["payload"]
    assert seen["url"] == "http://x:1/api/chat"
    assert p["model"] == "qwen3:latest" and p["stream"] is False and p["think"] is True
    assert p["format"] == SCHEMA
    assert p["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 99}
    assert [m["role"] for m in p["messages"]] == ["system", "user"]


def test_thinking_is_off_unless_asked_for(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "post",
                        lambda url, json=None, timeout=None: seen.update(p=json) or reply('{"a": 1}'))
    oj.chat_json("m", "s", "u", SCHEMA)
    assert seen["p"]["think"] is False


def test_chat_json_handles_reasoning_written_inside_the_answer(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: reply('<think>long reasoning</think>{"a": 2}'))
    assert oj.chat_json("m", "s", "u", SCHEMA)[0] == {"a": 2}


def test_cut_off_answer_says_so(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: reply('{"a": ', done_reason="length"))
    with pytest.raises(oj.LLMError) as err:
        oj.chat_json("m", "s", "u", SCHEMA)
    assert "cut off" in str(err.value) and not isinstance(err.value, oj.OllamaUnavailable)


def test_missing_model_is_fatal_not_retryable(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status=404))
    with pytest.raises(oj.OllamaUnavailable) as err:
        oj.chat_json("nope", "s", "u", SCHEMA)
    assert "ollama pull nope" in str(err.value)


def test_ollama_not_running_is_fatal(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(oj.OllamaUnavailable):
        oj.chat_json("m", "s", "u", SCHEMA)


def test_timeout_and_server_errors_are_retryable(monkeypatch):
    def slow(*a, **k):
        raise requests.exceptions.ReadTimeout()
    monkeypatch.setattr(requests, "post", slow)
    with pytest.raises(oj.LLMError) as err:
        oj.chat_json("m", "s", "u", SCHEMA, timeout=5)
    assert not isinstance(err.value, oj.OllamaUnavailable)

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status=500))
    with pytest.raises(oj.LLMError) as err:
        oj.chat_json("m", "s", "u", SCHEMA)
    assert not isinstance(err.value, oj.OllamaUnavailable)
