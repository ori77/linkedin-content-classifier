"""
A small, reusable helper: ask a local Ollama model a question and get back a
Python dictionary that is guaranteed to follow a JSON shape you define.

Used by step 5 (naming clusters) and later by the real 7-lens classifier.
"""

import json
import re
import time

import requests

OLLAMA_URL = "http://127.0.0.1:11434"


class LLMError(Exception):
    """The model answered, but the answer was unusable. Trying again may help."""


class OllamaUnavailable(LLMError):
    """Ollama is not running, or does not have the model. Trying again will not help."""


def strip_thinking(text):
    """Some Ollama versions put the model's reasoning inside <think>...</think> in the
    answer itself. Remove it so only the final answer is left."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat_json(model, system, user, schema, url=OLLAMA_URL, num_ctx=8192,
              think=False, num_predict=1500, timeout=1800, keep_alive="30m"):
    """Send one question. Returns (answer_dict, stats_dict).

    Raises OllamaUnavailable if Ollama or the model is missing,
    and LLMError if the answer is cut off or is not valid JSON.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": think,
        "format": schema,                  # Ollama forces the answer to follow this shape
        "options": {
            "temperature": 0,              # same input -> same answer
            "num_ctx": num_ctx,            # working memory, in tokens
            "num_predict": num_predict,    # hard cap on how many tokens it may write
        },
        "keep_alive": keep_alive,
    }

    started = time.time()
    try:
        response = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailable("Cannot reach Ollama. Is it running? Try: ollama ps")
    except requests.exceptions.Timeout:
        raise LLMError(f"No answer within {timeout} seconds")

    if response.status_code == 404:
        raise OllamaUnavailable(f"Ollama does not have '{model}'. Run: ollama pull {model}")
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise LLMError(f"Ollama returned an error: {e}")

    data = response.json()
    content = strip_thinking(data.get("message", {}).get("content", ""))
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        why = data.get("done_reason", "unknown")
        hint = " (the answer was cut off: raise num_predict)" if why == "length" else ""
        raise LLMError(f"Answer was not valid JSON, stop reason: {why}{hint}")

    stats = {
        "seconds": round(time.time() - started, 1),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }
    return answer, stats
