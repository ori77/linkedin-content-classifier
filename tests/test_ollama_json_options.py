"""Tests for the optional `options` argument of ollama_json.chat_json (Step 9)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import ollama_json as O  # noqa: E402


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps({"ok": True})}, "done_reason": "stop",
                "prompt_eval_count": 5, "eval_count": 3}


@pytest.fixture
def sent(monkeypatch):
    box = {}

    def fake_post(url, json=None, timeout=None):
        box["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(O.requests, "post", fake_post)
    return box


def test_without_options_the_request_is_unchanged(sent):
    O.chat_json("m", "s", "u", {"type": "object"})
    assert sent["payload"]["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 1500}


def test_extra_options_are_added(sent):
    O.chat_json("m", "s", "u", {"type": "object"}, options={"presence_penalty": 0})
    opts = sent["payload"]["options"]
    assert opts["presence_penalty"] == 0
    assert opts["temperature"] == 0 and opts["num_ctx"] == 8192     # the others stay


def test_extra_options_can_override_a_default(sent):
    O.chat_json("m", "s", "u", {"type": "object"}, options={"temperature": 0.3})
    assert sent["payload"]["options"]["temperature"] == 0.3


def test_empty_options_dict_changes_nothing(sent):
    O.chat_json("m", "s", "u", {"type": "object"}, options={})
    assert sent["payload"]["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 1500}
