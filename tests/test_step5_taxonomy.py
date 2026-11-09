"""Tests for step5_taxonomy.py (v2). A fake LLM replaces Ollama, so nothing here needs it."""
import csv
import json
import re

import numpy as np
import pytest
import yaml

pytest.importorskip("umap")
import step5_taxonomy as s5
from ollama_json import LLMError, OllamaUnavailable


# ------------------------------------------------------------------ a tiny invented world
def make_world(tmp_path, n_clusters=10, per=5, noise=5, dim=16):
    """Writes embeddings, meta and a clusters CSV for n clusters of `per` posts + noise."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(n_clusters, dim))
    rows, vectors, labels, ids = [], [], [], []
    for c in range(n_clusters):
        for _ in range(per):
            v = centres[c] + rng.normal(size=dim) * 0.1
            vectors.append(v / np.linalg.norm(v)); labels.append(c)
    for _ in range(noise):
        v = rng.normal(size=dim); vectors.append(v / np.linalg.norm(v)); labels.append(-1)
    for i, label in enumerate(labels):
        ids.append(str(1000 + i))
        used = [f"subject{label} alpha", f"subject{label} beta", "extra"] if label >= 0 \
            else [f"random{i}", "misc"]
        rows.append({"activity_id": ids[-1], "cluster": label,
                     "stage": "noise" if label < 0 else "cluster",
                     "keyphrases": "; ".join(used), "keyphrases_used": "; ".join(used)})
    np.save(tmp_path / "emb.npy", np.array(vectors, dtype=np.float32))
    json.dump({"activity_ids": ids}, open(tmp_path / "emb_meta.json", "w"))
    with open(tmp_path / "clusters.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return ids, np.array(labels)


class FakeLLM:
    """Stands in for chat_json.
    naming   : names cluster N 'SubjectN Stuff.'  (or 'Same Name' for clusters 1 and 2 if dup_names)
    proposing: returns `n_topics` topics called 'Topic 0'..
    placing  : cluster N goes to topic number N % (number of real topics); to General if N in general_for
    """
    def __init__(self, n_topics=8, dup_names=False, propose=None, general_for=(), fail_place=()):
        self.calls = {"naming": 0, "propose": 0, "place": 0}
        self.models, self.prompts = {"naming": set(), "propose": set(), "place": set()}, []
        self.n_topics, self.dup_names, self.propose = n_topics, dup_names, propose
        self.general_for, self.fail_place = set(general_for), set(fail_place)

    def __call__(self, model, system, user, schema, **kwargs):
        self.prompts.append(user)
        stats = {"seconds": 0.1, "prompt_tokens": 1, "output_tokens": 1}
        if schema is s5.NAMING_SHAPE:
            self.calls["naming"] += 1; self.models["naming"].add(model)
            word = re.search(r"Top words: ([^,\n]+)", user).group(1)
            n = int(re.search(r"(\d+)", word).group(1))
            name = "Same Name" if self.dup_names and n in (1, 2) else f"{word.title()} Stuff."
            return {"name": name, "description": f"About {word}.",
                    "coherence": "generic" if n == 0 else "clear", "reason": "because"}, stats
        if schema is s5.PROPOSE_SHAPE:
            self.calls["propose"] += 1; self.models["propose"].add(model)
            if self.propose:
                return self.propose(self.calls["propose"]), stats
            return {"topics": [{"name": f"Topic {i}", "description": f"Topic {i} stuff"}
                               for i in range(self.n_topics)]}, stats
        # placing one cluster
        self.calls["place"] += 1; self.models["place"].add(model)
        n = int(re.search(r"Name: Subject(\d+)|Name: .*?(\d+)", user).group(1) or 0) \
            if re.search(r"Name: Subject(\d+)", user) else -1
        if n in self.fail_place:
            raise LLMError("garbage")
        names = schema["properties"]["topic"]["enum"]
        real = names[:-1]
        choice = s5.GENERAL_TOPIC["name"] if n in self.general_for else real[max(n, 0) % len(real)]
        return {"topic": choice, "reason": "fits"}, stats


def run_main(monkeypatch, tmp_path, fake, extra=()):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s5, "chat_json", fake)
    monkeypatch.setattr("sys.argv", ["step5", "clusters.csv", "--emb-prefix", "emb", *extra])
    s5.main()


def load_doc(tmp_path, prefix="taxonomy"):
    return yaml.safe_load(open(tmp_path / "taxonomy" / f"{prefix}.draft.yaml", encoding="utf-8"))


# --------------------------------------------------------------------- small pure functions
def test_slugify_and_unique_slug():
    assert s5.slugify("RAG & Retrieval!") == "rag-retrieval"
    assert s5.slugify("???") == "untitled"
    used = set()
    assert [s5.unique_slug("a", used) for _ in range(3)] == ["a", "a-2", "a-3"]


def test_pick_examples_takes_the_central_ones_then_spreads():
    chosen = s5.pick_examples(list(range(20)))
    assert chosen[:8] == list(range(8)) and len(chosen) == 12 and len(set(chosen)) == 12
    assert chosen[-1] == 19
    assert s5.pick_examples(list(range(5))) == list(range(5))
    assert len(s5.pick_examples(list(range(10)))) == 10


def test_cache_key_changes_when_anything_that_matters_changes():
    base = s5.cache_key("v1", "qwen3", False, "prompt")
    assert base == s5.cache_key("v1", "qwen3", False, "prompt")
    for changed in [("v2", "qwen3", False, "prompt"), ("v1", "gemma", False, "prompt"),
                    ("v1", "qwen3", True, "prompt"), ("v1", "qwen3", False, "prompt2")]:
        assert s5.cache_key(*changed) != base


def test_tidy_name_answer_cleans_and_rejects():
    good = s5.tidy_name_answer({"name": ' "Vector  Databases." ', "description": " d  e ",
                                "coherence": "clear", "reason": "r"})
    assert good["name"] == "Vector Databases" and good["description"] == "d e"
    for bad in [{"name": "", "description": "d"}, {"name": "x" * 61, "description": "d"},
                {"name": "ok", "description": ""}]:
        with pytest.raises(LLMError):
            s5.tidy_name_answer(bad)


def test_an_invented_coherence_label_becomes_mixed():
    out = s5.tidy_name_answer({"name": "Ok", "description": "d", "coherence": "very clear"})
    assert out["coherence"] == "mixed"


def test_cache_ignores_a_half_written_line_and_append_repairs_it(tmp_path):
    path = str(tmp_path / "n.jsonl")
    s5.append_jsonl(path, {"cluster": 1, "key": "k", "result": {}})
    with open(path, "a") as f:
        f.write('{"cluster": 2, "ke')
    assert list(s5.load_name_cache(path)) == [(1, "k")]
    s5.append_jsonl(path, {"cluster": 3, "key": "k", "result": {}})
    assert sorted(s5.load_name_cache(path)) == [(1, "k"), (3, "k")]


# ------------------------------------------------------------------------- the topic proposal
def topics_raw(n):
    return {"topics": [{"name": f"T{i}", "description": "d"} for i in range(n)]}


def test_validate_topics_accepts_a_good_list():
    topics, problems = s5.validate_topics(topics_raw(9))
    assert len(topics) == 9 and problems == []


def test_validate_topics_drops_duplicates_blanks_and_the_reserved_general_topic():
    raw = {"topics": [{"name": "Alpha", "description": "d"}, {"name": " alpha ", "description": "d"},
                      {"name": "", "description": "d"}, {"name": "Beta", "description": ""},
                      {"name": "general ai engineering", "description": "d"},
                      {"name": "Gamma.", "description": "d"}]}
    topics, _ = s5.validate_topics(raw)
    assert [t["name"] for t in topics] == ["Alpha", "Gamma"]


def test_validate_topics_flags_a_wrong_number():
    assert s5.validate_topics(topics_raw(3))[1] and s5.validate_topics(topics_raw(20))[1]


def test_assign_shape_only_allows_the_listed_topics():
    shape = s5.assign_shape(["A", "B"])
    assert shape["properties"]["topic"]["enum"] == ["A", "B"]


def test_place_cluster_rejects_a_topic_that_is_not_on_the_list(monkeypatch):
    class Args: model = "m"; url = "u"; num_ctx = 1; think = False; max_tokens = 5
    monkeypatch.setattr(s5, "chat_json", lambda *a, **k: ({"topic": "Invented", "reason": "r"}, {}))
    answer, error = s5.place_cluster("p", ["A", "B"], Args())
    assert answer is None and "unknown topic" in error


# ------------------------------------------------------------------ measuring generic clusters
def test_disambiguate_names_tells_identical_names_apart():
    named = {0: {"name": "AI Practices"}, 1: {"name": "AI Practices"}, 2: {"name": "AI Practices"},
             3: {"name": "Unique One"}}
    cards = {0: {"top_words": ["code", "review"]}, 1: {"top_words": ["langchain", "graph"]},
             2: {"top_words": ["code", "review"]}, 3: {"top_words": ["x"]}}
    out = s5.disambiguate_names(named, cards)
    names = [out[c]["name"] for c in sorted(out)]
    assert len({n.lower() for n in names}) == 4
    assert names[3] == "Unique One" and names[1] == "AI Practices (langchain, graph)"


def test_cluster_metrics_marks_the_broad_loose_cluster():
    rng = np.random.default_rng(1)
    labels, seen, vecs = [], [], []
    for c in range(4):                                   # four specific, tight clusters
        centre = rng.normal(size=8)
        for _ in range(6):
            v = centre + rng.normal(size=8) * 0.05
            vecs.append(v / np.linalg.norm(v)); labels.append(c)
            seen.append([f"topic{c} detail", "ai llm"])
    for _ in range(6):                                   # one loose cluster of broad wording
        v = rng.normal(size=8); vecs.append(v / np.linalg.norm(v)); labels.append(4)
        seen.append(["ai llm", "ai llm"])
    m = s5.cluster_metrics(np.array(labels), seen, np.array(vecs))
    assert m[4]["generic_share"] > max(m[c]["generic_share"] for c in range(4))
    assert m[4]["tightness"] < min(m[c]["tightness"] for c in range(4))
    assert s5.flag_generic(m, fraction=0.2) == {4}


def test_flag_generic_uses_wording_only_not_tightness():
    # a very TIGHT cluster with broad wording is flagged; a LOOSE cluster with specific wording is not
    metrics = {0: {"tightness": 0.97, "generic_share": 0.9},
               1: {"tightness": 0.80, "generic_share": 0.05},
               2: {"tightness": 0.90, "generic_share": 0.3},
               3: {"tightness": 0.92, "generic_share": 0.4}}
    assert s5.flag_generic(metrics, fraction=0.25) == {0}


def test_prompts_carry_the_new_rules():
    assert "must NOT overlap" in s5.PROPOSE_SYSTEM and "Practices" in s5.PROPOSE_SYSTEM
    assert "MOST SPECIFIC" in s5.ASSIGN_SYSTEM
    card = {"top_words": ["w"], "examples": [["a"]]}
    hint = s5.assignment_prompt([{"name": "T", "description": "d"}], card,
                                {"name": "N", "description": "D", "size": 1, "looks_generic": True})
    assert "only a hint" in hint


def test_flag_generic_picks_the_requested_share_and_handles_tiny_input():
    metrics = {c: {"tightness": 0.9 - c * 0.01, "generic_share": c / 10} for c in range(8)}
    assert s5.flag_generic(metrics, fraction=0.25) == {6, 7}
    assert s5.flag_generic({0: {"tightness": 1, "generic_share": 0}}) == set()


def test_assignment_prompt_lists_topics_and_shows_the_hint_only_when_flagged():
    card = {"top_words": ["w"], "examples": [["a", "b"], ["c"], ["d"], ["e"]]}
    cluster = {"name": "N", "description": "D", "size": 3, "looks_generic": False}
    topics = [{"name": "T1", "description": "one"}, {"name": "T2", "description": "two"}]
    plain = s5.assignment_prompt(topics, card, cluster)
    assert "- T1: one" in plain and "- T2: two" in plain and "Measured hint" not in plain
    assert plain.count("\n1. ") == 1 and "4. " not in plain          # only 3 examples
    assert "Measured hint" in s5.assignment_prompt(topics, card, {**cluster, "looks_generic": True})


# ------------------------------------------------------------------- build_taxonomy / loading
def test_build_taxonomy_makes_unique_ids_sorts_by_size_and_keeps_unassigned_last():
    named = {0: {"name": "Same", "size": 5}, 1: {"name": "Same", "size": 9},
             2: {"name": "Other One", "size": 20}, 3: {"name": "Lost", "size": 99}}
    for n in named.values():
        n.update(description="d", coherence="clear", reason="r")
    cards = {c: {"top_words": ["w"], "examples": []} for c in named}
    topics = [{"name": "Alpha", "description": "d", "cluster_ids": [0, 1]},
              {"name": "Alpha", "description": "d", "cluster_ids": [2]},
              {"name": s5.UNASSIGNED, "description": "d", "cluster_ids": [3]}]
    tax = s5.build_taxonomy(topics, named, cards, noise_count=4)
    assert tax["topics"][-1]["name"] == s5.UNASSIGNED               # last even though biggest
    assert tax["topics"][0]["post_count"] == 20
    assert len({t["id"] for t in tax["topics"]}) == 3
    assert [s["cluster"] for s in tax["topics"][1]["subtopics"]] == [1, 0]
    assert tax["other"]["post_count"] == 4


def test_load_clusters_stops_when_posts_do_not_match(tmp_path):
    make_world(tmp_path)
    meta = json.load(open(tmp_path / "emb_meta.json"))
    meta["activity_ids"].reverse()
    json.dump(meta, open(tmp_path / "emb_meta.json", "w"))
    with pytest.raises(SystemExit):
        s5.load_clusters(tmp_path / "clusters.csv", str(tmp_path / "emb"))


# --------------------------------------------------------------------- the whole script
def test_full_run_writes_every_output_and_accounts_for_every_post(tmp_path, monkeypatch):
    ids, labels = make_world(tmp_path)
    fake = FakeLLM()
    run_main(monkeypatch, tmp_path, fake)
    doc = load_doc(tmp_path)
    assert len(doc["topics"]) == 8 and doc["other"]["post_count"] == 5
    assert sum(len(t["subtopics"]) for t in doc["topics"]) == 10          # every cluster placed once
    assert "Unassigned (needs review)" not in [t["name"] for t in doc["topics"]]
    sub = doc["topics"][0]["subtopics"][0]
    assert {"tightness", "generic_share", "looks_generic", "model_coherence"} <= set(sub)
    rows = list(csv.DictReader(open(tmp_path / "taxonomy" / "taxonomy_assignments.csv",
                                    encoding="utf-8-sig")))
    assert len(rows) == len(ids) and sum(r["subtopic_id"] == "other.general" for r in rows) == 5
    review = open(tmp_path / "taxonomy" / "taxonomy_review.md", encoding="utf-8").read()
    assert "Most generic-looking clusters" in review and "Other / General (5 posts)" in review
    assert fake.calls == {"naming": 10, "propose": 1, "place": 10}
    assert sum(s["looks_generic"] for t in doc["topics"] for s in t["subtopics"]) == 2   # 25% of 10


def test_a_second_run_reuses_saved_answers_but_changing_think_does_not(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM())
    again = FakeLLM()
    run_main(monkeypatch, tmp_path, again)
    assert again.calls == {"naming": 0, "propose": 0, "place": 0}
    thinking = FakeLLM()
    run_main(monkeypatch, tmp_path, thinking, extra=["--think"])
    assert thinking.calls == {"naming": 10, "propose": 1, "place": 10}


def test_a_new_topic_list_replaces_placements_but_keeps_cluster_names(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(n_topics=8))
    other = FakeLLM(n_topics=9)
    run_main(monkeypatch, tmp_path, other,
             extra=["--out-prefix", "second", "--topic-model", "bigger-model"])
    assert other.calls == {"naming": 0, "propose": 1, "place": 10}
    assert len(load_doc(tmp_path, "second")["topics"]) == 9
    assert len(load_doc(tmp_path)["topics"]) == 8                          # the first draft is untouched


def test_a_stopped_run_continues_where_it_left_off(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(), extra=["--only", "0,1,2"])
    assert not (tmp_path / "taxonomy" / "taxonomy.draft.yaml").exists()
    rest = FakeLLM()
    run_main(monkeypatch, tmp_path, rest)
    assert rest.calls["naming"] == 7


def test_topic_model_is_used_only_for_the_topic_list(tmp_path, monkeypatch):
    make_world(tmp_path)
    fake = FakeLLM()
    run_main(monkeypatch, tmp_path, fake, extra=["--model", "small", "--topic-model", "big"])
    assert fake.models == {"naming": {"small"}, "propose": {"big"}, "place": {"small"}}


def test_the_general_topic_appears_only_when_a_cluster_is_placed_there(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(general_for=[3, 4]))
    doc = load_doc(tmp_path)
    general = [t for t in doc["topics"] if t["name"] == "General AI Engineering"]
    assert len(general) == 1 and len(general[0]["subtopics"]) == 2


def test_identical_cluster_names_are_made_distinct(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(dup_names=True))
    doc = load_doc(tmp_path)
    names = [s["name"].lower() for t in doc["topics"] for s in t["subtopics"]]
    assert len(names) == len(set(names)) == 10


def test_topic_list_is_retried_once_with_the_problem_spelled_out(tmp_path, monkeypatch):
    make_world(tmp_path)
    fake = FakeLLM(propose=lambda call: topics_raw(3) if call == 1 else topics_raw(8))
    run_main(monkeypatch, tmp_path, fake)
    assert fake.calls["propose"] == 2
    prompt_of_retry = [p for p in fake.prompts if "previous answer had problems" in p]
    assert len(prompt_of_retry) == 1 and "between 7 and 11" in prompt_of_retry[0]
    assert len(load_doc(tmp_path)["topics"]) == 8


def test_a_topic_list_that_stays_wrong_still_finishes_with_a_warning(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(propose=lambda call: topics_raw(3)))
    review = open(tmp_path / "taxonomy" / "taxonomy_review.md", encoding="utf-8").read()
    assert "outside the 8-15 target" in review
    doc = load_doc(tmp_path)
    assert sum(len(t["subtopics"]) for t in doc["topics"]) == 10          # still nothing lost


def test_an_unusable_topic_list_stops_with_a_clear_message(tmp_path, monkeypatch):
    make_world(tmp_path)
    with pytest.raises(SystemExit) as err:
        run_main(monkeypatch, tmp_path, FakeLLM(propose=lambda call: {"topics": []}))
    assert "usable topic list" in str(err.value)


def test_a_cluster_that_cannot_be_placed_goes_to_unassigned_and_is_not_cached(tmp_path, monkeypatch):
    make_world(tmp_path)
    run_main(monkeypatch, tmp_path, FakeLLM(fail_place=[4]))
    doc = load_doc(tmp_path)
    assert doc["topics"][-1]["name"] == "Unassigned (needs review)"
    assert [s["cluster"] for s in doc["topics"][-1]["subtopics"]] == [4]
    cached = [json.loads(l)["cluster"] for l in
              open(tmp_path / "taxonomy" / "topic_assignments.jsonl", encoding="utf-8")]
    assert 4 not in cached and len(cached) == 9
    review = open(tmp_path / "taxonomy" / "taxonomy_review.md", encoding="utf-8").read()
    assert "could not be placed" in review


def test_naming_failures_get_a_placeholder_and_are_not_saved(tmp_path, monkeypatch):
    make_world(tmp_path, n_clusters=3, per=5, noise=0)

    def broken(*a, **k):
        raise LLMError("garbage")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s5, "chat_json", broken)
    monkeypatch.setattr("sys.argv", ["step5", "clusters.csv", "--emb-prefix", "emb", "--only", "0,1"])
    s5.main()
    saved = tmp_path / "taxonomy" / "cluster_names.jsonl"
    assert not saved.exists() or saved.read_text() == ""


def test_ollama_down_stops_the_run_with_a_clear_message(tmp_path, monkeypatch):
    make_world(tmp_path)

    def down(*a, **k):
        raise OllamaUnavailable("Cannot reach Ollama. Is it running?")
    with pytest.raises(SystemExit) as err:
        run_main(monkeypatch, tmp_path, down)
    assert "Is it running" in str(err.value)