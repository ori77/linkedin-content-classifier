"""Tests for step3_embeddings.py. None of them need Ollama to be running."""
import json

import numpy as np
import pytest
import requests

import step3_embeddings as s3


# ---------------------------------------------------------------- build_text
def test_build_text_adds_task_prefix_and_joins_phrases():
    assert s3.build_text(["rag evaluation", "llm agents"]) == \
        "clustering: rag evaluation, llm agents"


def test_build_text_removes_years():
    assert s3.build_text(["2026 travel trends", "ai skills for 2026"]) == \
        "clustering: travel trends, ai skills for"


def test_build_text_drops_phrase_that_was_only_a_year():
    assert s3.build_text(["rag", "2026"]) == "clustering: rag"


def test_build_text_keeps_numbers_that_are_not_years():
    assert s3.build_text(["gpt4 vs 100k tokens", "top 10 tips"]) == \
        "clustering: gpt4 vs 100k tokens, top 10 tips"


# ----------------------------------------------------------- load_keyphrases
def write_lines(path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_keyphrases_skips_a_half_written_last_line(tmp_path):
    f = tmp_path / "kp.jsonl"
    write_lines(f, [json.dumps({"activity_id": "1", "keyphrases": ["a", "b", "c"]}),
                    '{"activity_id": "2", "keyph'])
    assert s3.load_keyphrases(f) == {"1": ["a", "b", "c"]}


def test_load_keyphrases_later_duplicate_wins_and_order_is_kept(tmp_path):
    f = tmp_path / "kp.jsonl"
    write_lines(f, [json.dumps({"activity_id": "1", "keyphrases": ["old"]}),
                    json.dumps({"activity_id": "2", "keyphrases": ["x"]}),
                    json.dumps({"activity_id": "1", "keyphrases": ["new"]})])
    result = s3.load_keyphrases(f)
    assert result["1"] == ["new"]
    assert list(result) == ["1", "2"]


# ----------------------------------------------------------------- normalise
def test_normalise_makes_every_vector_length_one():
    out = s3.normalise(np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalise_leaves_zero_vector_as_zeros_not_nan():
    out = s3.normalise(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all() and not out.any()


# ----------------------------------------------------------- validate_vectors
def test_validate_accepts_good_vectors():
    s3.validate_vectors(s3.normalise(np.random.rand(5, 8).astype(np.float32)), list("abcde"))


def test_validate_rejects_nan():
    v = np.ones((2, 4), dtype=np.float32)
    v[0, 0] = np.nan
    with pytest.raises(SystemExit):
        s3.validate_vectors(v, ["a", "b"])


def test_validate_rejects_empty_vector_and_names_the_post(capsys):
    v = s3.normalise(np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32))
    with pytest.raises(SystemExit) as err:
        s3.validate_vectors(v, ["post-1", "post-2"])
    assert "post-2" in str(err.value)


# --------------------------------------------------------------- embed_batch
class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._body


def test_embed_batch_returns_one_row_per_text(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        body={"embeddings": [[1, 0, 0], [0, 1, 0]]}))
    out = s3.embed_batch(["a", "b"], "m", "http://x")
    assert out.shape == (2, 3) and out.dtype == np.float32


def test_embed_batch_explains_missing_model(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status=404))
    with pytest.raises(SystemExit) as err:
        s3.embed_batch(["a"], "nope", "http://x")
    assert "ollama pull nope" in str(err.value)


def test_embed_batch_explains_ollama_not_running(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(SystemExit) as err:
        s3.embed_batch(["a"], "m", "http://x")
    assert "Is it running" in str(err.value)


def test_embed_batch_catches_wrong_number_of_vectors(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        body={"embeddings": [[1, 0]]}))
    with pytest.raises(SystemExit):
        s3.embed_batch(["a", "b"], "m", "http://x")


# ------------------------------------------------------- the whole script
def test_main_end_to_end_with_limit(tmp_path, monkeypatch):
    kp = tmp_path / "kp.jsonl"
    write_lines(kp, [json.dumps({"activity_id": str(i), "keyphrases": [f"topic {i}", "x", "y"]})
                     for i in range(5)])

    def fake_embed(texts, model, url):
        rng = np.random.default_rng(len(texts))
        return rng.random((len(texts), 6)).astype(np.float32) + 0.1

    monkeypatch.setattr(s3, "embed_batch", fake_embed)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["step3", str(kp), "--limit", "3"])
    s3.main()

    vectors = np.load(tmp_path / "embeddings.npy")
    meta = json.loads((tmp_path / "embeddings_meta.json").read_text(encoding="utf-8"))
    assert vectors.shape == (3, 6)
    assert meta["activity_ids"] == ["0", "1", "2"]      # row i belongs to post i
    assert meta["task_prefix"] == "clustering: "
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
