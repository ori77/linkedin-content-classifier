"""Tests for step4_cluster.py. They use small invented data and never need Ollama."""
import csv
import json

import numpy as np
import pytest

pytest.importorskip("umap")
pytest.importorskip("sklearn")
import step4_cluster as s4


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def blob(rng, centre, n, spread=0.6):
    return unit(centre + rng.normal(size=(n, len(centre))) * spread)


# ---------------------------------------------------------------- words_of
def test_words_of_drops_filler_and_splits_hyphens():
    assert s4.words_of(["rag for the llm", "fine-tuning of models"]) == \
        ["rag", "llm", "fine", "tuning", "models"]


def test_words_of_drops_one_letter_words():
    assert s4.words_of(["a b rag"]) == ["rag"]


# ------------------------------------------------------ top_words_per_cluster
def test_top_words_prefer_words_that_are_special_to_the_cluster():
    labels = np.array([0, 0, 0, 1, 1, 1])
    words = [["salary", "ai"], ["salary", "ai"], ["salary", "ai"],
             ["rag", "ai"], ["rag", "ai"], ["rag", "ai"]]
    top = s4.top_words_per_cluster(labels, words, how_many=2)
    assert top[0][0] == "salary" and top[1][0] == "rag"     # 'ai' is everywhere, so it ranks lower


def test_top_words_ignore_noise_label():
    labels = np.array([0, 0, -1])
    top = s4.top_words_per_cluster(labels, [["a1"], ["a1"], ["junk"]], how_many=3)
    assert set(top) == {0} and "junk" not in top[0]


# ----------------------------------------------------------- representatives
def test_representatives_returns_the_posts_nearest_the_centre():
    vectors = unit([[1, 0], [0.99, 0.1], [0.9, 0.5], [0, 1]])
    labels = np.array([0, 0, 0, 1])
    rows = s4.representatives(vectors, labels, cluster=0, how_many=2)
    assert len(rows) == 2 and 2 not in rows and 3 not in rows   # the stray one is not typical


# ----------------------------------------------------------- peel_outliers
def test_peel_removes_a_stray_post_and_keeps_the_rest():
    rng = np.random.default_rng(0)
    centre = rng.normal(size=64)
    vectors = np.vstack([blob(rng, centre, 8), unit(rng.normal(size=(2, 64)))])
    kept = s4.peel_outliers(vectors, np.arange(10), bar=0.7, min_keep=6)
    assert set(kept) == set(range(8))


def test_peel_gives_up_when_too_few_posts_are_typical():
    rng = np.random.default_rng(0)
    vectors = unit(rng.normal(size=(10, 64)))                  # unrelated posts
    assert len(s4.peel_outliers(vectors, np.arange(10), bar=0.7, min_keep=6)) == 0


# ------------------------------------------------------------- rescue_noise
def make_world(seed):
    rng = np.random.default_rng(seed)
    c = [rng.normal(size=768) for _ in range(3)]
    vectors = np.vstack([blob(rng, c[0], 40), blob(rng, c[1], 8), blob(rng, c[2], 9),
                         unit(rng.normal(size=(40, 768)))])
    labels = np.array([0] * 40 + [-1] * 57)        # first pass found only the big topic
    return vectors, labels


def test_rescue_finds_small_topics_and_leaves_junk_and_old_clusters_alone():
    vectors, labels = make_world(seed=2)
    new, rescued = s4.rescue_noise(vectors, labels, 15, 10, 42, 6, 3, "leaf")
    assert (new[:40] == 0).all()                                 # first-pass cluster untouched
    assert len(rescued) == 2                                     # both small topics found
    a, b = set(new[40:48]) - {-1}, set(new[48:57]) - {-1}
    assert len(a) == 1 and len(b) == 1 and a != b                # each topic has its own label
    assert (new[57:] == -1).all()                                # random leftovers stay noise


def test_rescue_does_nothing_with_too_few_noise_posts():
    vectors = unit(np.random.default_rng(0).normal(size=(20, 16)))
    labels = np.array([0] * 5 + [-1] * 15)
    new, rescued = s4.rescue_noise(vectors, labels, 15, 10, 42, 6, 3, "leaf")
    assert rescued == set() and (new == labels).all()


def test_rescue_does_not_invent_clusters_from_pure_noise():
    rng = np.random.default_rng(3)
    vectors = np.vstack([blob(rng, rng.normal(size=768), 40), unit(rng.normal(size=(60, 768)))])
    labels = np.array([0] * 40 + [-1] * 60)
    new, rescued = s4.rescue_noise(vectors, labels, 15, 10, 42, 6, 3, "leaf")
    assert rescued == set() and (new[40:] == -1).all()


# ------------------------------------------------------------------ outputs
def test_stage_of():
    assert s4.stage_of(-1, {5}) == "noise"
    assert s4.stage_of(5, {5}) == "rescued"
    assert s4.stage_of(2, {5}) == "cluster"


def test_write_csv_has_both_phrase_versions_and_stage(tmp_path):
    path = tmp_path / "c.csv"
    s4.write_csv(path, ["1", "2"], np.array([3, -1]), {3},
                 {"1": ["rag interview"], "2": ["x"]}, {"1": ["rag"], "2": ["x"]})
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    assert rows[0] == {"activity_id": "1", "cluster": "3", "stage": "rescued",
                       "keyphrases": "rag interview", "keyphrases_used": "rag"}
    assert rows[1]["stage"] == "noise" and rows[1]["cluster"] == "-1"


# -------------------------------------------------------------- loading
def test_load_inputs_rebuilds_what_the_model_saw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.save("emb.npy", unit(np.ones((2, 4))))
    json.dump({"activity_ids": ["1", "2"], "dropped_words": ["interview"]},
              open("emb_meta.json", "w"))
    with open("kp.jsonl", "w") as f:
        f.write(json.dumps({"activity_id": "1", "keyphrases": ["rag interview", "2026 trends"]}) + "\n")
        f.write(json.dumps({"activity_id": "2", "keyphrases": ["agents"]}) + "\n")
    _, ids, original, seen = s4.load_inputs("emb", "kp.jsonl")
    assert original["1"] == ["rag interview", "2026 trends"]
    assert seen["1"] == ["rag", "trends"]                       # 'interview' and the year are gone


def test_load_inputs_stops_when_rows_and_ids_disagree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.save("emb.npy", unit(np.ones((3, 4))))
    json.dump({"activity_ids": ["1", "2"]}, open("emb_meta.json", "w"))
    open("kp.jsonl", "w").write("")
    with pytest.raises(SystemExit):
        s4.load_inputs("emb", "kp.jsonl")


# ------------------------------------------------------- whole script
def test_main_end_to_end_with_rescue(tmp_path, monkeypatch):
    vectors, _ = make_world(seed=2)
    ids = [str(i) for i in range(len(vectors))]
    monkeypatch.chdir(tmp_path)
    np.save("embeddings.npy", vectors)
    json.dump({"activity_ids": ids, "dropped_words": []}, open("embeddings_meta.json", "w"))
    with open("kp.jsonl", "w") as f:
        for i in ids:
            f.write(json.dumps({"activity_id": i, "keyphrases": [f"topic {int(i) // 10}", "x", "y"]}) + "\n")

    # Make the FIRST clustering pass deterministic: it finds only the big topic and leaves
    # everything else as noise. The second (rescue) pass, UMAP and all outputs run for real.
    real_find_clusters, calls = s4.find_clusters, {"n": 0}

    def first_pass_then_real(reduced, size, min_samples, method):
        calls["n"] += 1
        if calls["n"] == 1:
            return np.array([0] * 40 + [-1] * 57)
        return real_find_clusters(reduced, size, min_samples, method)

    monkeypatch.setattr(s4, "find_clusters", first_pass_then_real)
    monkeypatch.setattr("sys.argv", ["step4", "kp.jsonl", "--rescue-noise"])
    s4.main()

    rows = list(csv.DictReader(open("clusters.csv", encoding="utf-8-sig")))
    assert len(rows) == len(ids)
    assert {r["stage"] for r in rows} <= {"cluster", "rescued", "noise"}
    assert sum(r["stage"] == "rescued" for r in rows) >= 12      # the two small topics
    assert all(r["stage"] == "noise" for r in rows[57:])          # random leftovers stay noise
    report = open("clusters_report.md", encoding="utf-8").read()
    assert "rescued from noise" in report and "Unclustered" in report
