"""Tests for step5b_finalize.py: checking a hand-edited draft and freezing it as taxonomy v1."""
import csv
import json

import numpy as np
import pytest
import yaml

pytest.importorskip("umap")
import step5_taxonomy as s5
import step5b_finalize as fin


def make_world(tmp_path, n_clusters=6, per=5, noise=4, dim=8):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(n_clusters, dim))
    rows, vectors, ids = [], [], []
    labels = [c for c in range(n_clusters) for _ in range(per)] + [-1] * noise
    for label in labels:
        v = centres[label] + rng.normal(size=dim) * 0.1 if label >= 0 else rng.normal(size=dim)
        vectors.append(v / np.linalg.norm(v))
    for i, label in enumerate(labels):
        ids.append(str(2000 + i))
        used = [f"subject{label} alpha", f"subject{label} beta"] if label >= 0 else [f"junk{i}"]
        rows.append({"activity_id": ids[-1], "cluster": label,
                     "stage": "noise" if label < 0 else "cluster",
                     "keyphrases": "; ".join(used), "keyphrases_used": "; ".join(used)})
    np.save(tmp_path / "emb.npy", np.array(vectors, dtype=np.float32))
    json.dump({"activity_ids": ids}, open(tmp_path / "emb_meta.json", "w"))
    with open(tmp_path / "clusters.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return ids, np.array(labels)


def sub(name, *clusters, description="d"):
    key = {"cluster": clusters[0]} if len(clusters) == 1 else {"clusters": list(clusters)}
    return {"name": name, "description": description, **key}


def draft(topics=None, other=None):
    topics = topics if topics is not None else [
        {"name": "Alpha", "description": "d", "subtopics": [sub("S0", 0), sub("S1", 1)]},
        {"name": "Beta", "description": "d", "subtopics": [sub("S2", 2), sub("S3", 3)]},
        {"name": "Gamma", "description": "d", "subtopics": [sub("S4", 4), sub("S5", 5)]}]
    return {"topics": topics, **({"other": other} if other is not None else {})}


SIZES = {c: 5 for c in range(6)}
WORDS = {c: [f"w{c}"] for c in range(6)}


def run(doc, noise=4):
    return fin.finalize(doc, SIZES, WORDS, noise)


# --------------------------------------------------------------------------- the checks
def test_a_consistent_draft_passes_and_counts_are_recomputed():
    taxonomy, problems, _ = run(draft())
    assert problems == []
    assert [t["post_count"] for t in taxonomy["topics"]] == [10, 10, 10]
    assert taxonomy["other"]["post_count"] == 4
    assert taxonomy["topics"][0]["subtopics"][0]["id"].count(".") == 1


def test_ids_are_regenerated_from_names_and_stay_unique():
    topics = [{"name": "Same Name", "description": "d", "subtopics": [sub("A", 0), sub("B", 1)]},
              {"name": "same-name", "description": "d", "subtopics": [sub("C", 2), sub("D", 3)]},
              {"name": "Gamma", "description": "d", "subtopics": [sub("E", 4), sub("F", 5)]}]
    taxonomy, problems, _ = run(draft(topics))
    assert problems == []
    assert len({t["id"] for t in taxonomy["topics"]}) == 3


def test_moving_a_subtopic_to_another_topic_updates_the_counts():
    topics = [{"name": "Alpha", "description": "d", "subtopics": [sub("S0", 0)]},
              {"name": "Beta", "description": "d", "subtopics": [sub("S1", 1), sub("S2", 2), sub("S3", 3)]},
              {"name": "Gamma", "description": "d", "subtopics": [sub("S4", 4), sub("S5", 5)]}]
    taxonomy, problems, _ = run(draft(topics))
    assert problems == []
    assert {t["name"]: t["post_count"] for t in taxonomy["topics"]} == {"Alpha": 5, "Beta": 15, "Gamma": 10}


def test_a_brand_new_topic_needs_only_name_description_and_subtopics():
    topics = [{"name": "Alpha", "description": "d", "subtopics": [sub("S0", 0), sub("S1", 1), sub("S2", 2)]},
              {"name": "New Topic", "description": "fresh", "subtopics": [sub("S3", 3)]},
              {"name": "Gamma", "description": "d", "subtopics": [sub("S4", 4), sub("S5", 5)]}]
    taxonomy, problems, _ = run(draft(topics))
    assert problems == []
    new = [t for t in taxonomy["topics"] if t["name"] == "New Topic"][0]
    assert new["id"] == "new-topic" and new["subtopics"][0]["keywords"] == ["w3"]


def test_merging_clusters_into_one_subtopic_adds_their_sizes():
    topics = [{"name": "Alpha", "description": "d", "subtopics": [sub("Merged", 0, 1, 2)]},
              {"name": "Beta", "description": "d", "subtopics": [sub("S3", 3)]},
              {"name": "Gamma", "description": "d", "subtopics": [sub("S4", 4), sub("S5", 5)]}]
    taxonomy, problems, _ = run(draft(topics))
    merged = [s for t in taxonomy["topics"] for s in t["subtopics"] if s["name"] == "Merged"][0]
    assert problems == [] and merged["post_count"] == 15 and merged["clusters"] == [0, 1, 2]


def test_sending_clusters_to_other_adds_them_to_the_other_count():
    topics = [{"name": "Alpha", "description": "d", "subtopics": [sub("S0", 0), sub("S1", 1)]},
              {"name": "Beta", "description": "d", "subtopics": [sub("S2", 2), sub("S3", 3)]},
              {"name": "Gamma", "description": "d", "subtopics": [sub("S4", 4)]}]
    taxonomy, problems, _ = run(draft(topics, other={"clusters": [5]}))
    assert problems == [] and taxonomy["other"]["post_count"] == 4 + 5
    assert taxonomy["other"]["clusters"] == [5]


@pytest.mark.parametrize("mutate, expected", [
    (lambda t: t[0]["subtopics"].pop(), "not in any subtopic"),                       # a cluster forgotten
    (lambda t: t[0]["subtopics"].append(sub("Dup", 2)), "used twice"),                # a cluster used twice
    (lambda t: t[0]["subtopics"].append(sub("Ghost", 99)), "does not exist"),         # unknown cluster
    (lambda t: t[1].update(name="alpha"), "Two topics are called"),                   # duplicate topic name
    (lambda t: t[0]["subtopics"][1].update(name="S0"), "Two subtopics are called"),   # duplicate subtopic
    (lambda t: t[0].update(description=""), "has no description"),
    (lambda t: t[0]["subtopics"][0].update(description=" "), "has no description"),
    (lambda t: t[0].update(name="Unassigned (needs review)"), "must be resolved"),
    (lambda t: t[0]["subtopics"][0].update(cluster="five"), "unusable cluster ids"),
])
def test_problems_are_reported_in_plain_words(mutate, expected):
    topics = draft()["topics"]
    mutate(topics)
    _, problems, _ = run({"topics": topics})
    assert any(expected in p for p in problems), problems


def test_an_empty_topic_is_a_problem_and_a_wrong_count_is_only_a_warning():
    topics = draft()["topics"] + [{"name": "Empty", "description": "d", "subtopics": []}]
    _, problems, warnings_ = run({"topics": topics})
    assert any("no subtopics" in p for p in problems)
    assert any("topics (the usual target" in w for w in warnings_)


def test_a_file_that_is_not_a_draft_is_rejected_politely():
    assert fin.finalize({"nothing": 1}, SIZES, WORDS, 0)[1]
    assert fin.finalize(None, SIZES, WORDS, 0)[1]


# --------------------------------------------------------------------------- the script
def write_yaml(tmp_path, doc, name="edited.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return str(path)


def run_main(monkeypatch, tmp_path, yaml_path, extra=()):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["step5b", yaml_path, "clusters.csv", "--emb-prefix", "emb", *extra])
    fin.main()


def test_main_writes_the_frozen_taxonomy_and_assignments(tmp_path, monkeypatch):
    ids, labels = make_world(tmp_path)
    run_main(monkeypatch, tmp_path, write_yaml(tmp_path, draft()))
    frozen = yaml.safe_load(open(tmp_path / "taxonomy" / "taxonomy.v1.yaml", encoding="utf-8"))
    assert frozen["version"] == "v1" and frozen["based_on"] == "edited.yaml"
    assert len(frozen["topics"]) == 3 and frozen["other"]["post_count"] == 4
    rows = list(csv.DictReader(open(tmp_path / "taxonomy" / "taxonomy_assignments.v1.csv",
                                    encoding="utf-8-sig")))
    assert len(rows) == len(ids)
    assert sum(r["subtopic_id"] == "other.general" for r in rows) == 4
    assert {r["subtopic_id"] for r in rows if r["cluster"] == "0"} == {"alpha.s0"}


def test_main_stops_with_the_list_of_problems_and_writes_nothing(tmp_path, monkeypatch, capsys):
    make_world(tmp_path)
    bad = draft()
    bad["topics"][0]["subtopics"].pop()
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, write_yaml(tmp_path, bad))
    assert "not in any subtopic" in capsys.readouterr().out
    assert not (tmp_path / "taxonomy" / "taxonomy.v1.yaml").exists()


def test_main_will_not_overwrite_a_frozen_taxonomy_without_force(tmp_path, monkeypatch):
    make_world(tmp_path)
    path = write_yaml(tmp_path, draft())
    run_main(monkeypatch, tmp_path, path)
    with pytest.raises(SystemExit) as err:
        run_main(monkeypatch, tmp_path, path)
    assert "already exists" in str(err.value)
    run_main(monkeypatch, tmp_path, path, extra=["--force"])                 # allowed when asked
    run_main(monkeypatch, tmp_path, path, extra=["--version", "v2"])         # or as a new version
    assert (tmp_path / "taxonomy" / "taxonomy.v2.yaml").exists()


def test_main_stops_when_the_posts_do_not_match_the_embeddings(tmp_path, monkeypatch):
    make_world(tmp_path)
    meta = json.load(open(tmp_path / "emb_meta.json"))
    meta["activity_ids"].reverse()
    json.dump(meta, open(tmp_path / "emb_meta.json", "w"))
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, write_yaml(tmp_path, draft()))


def test_an_unedited_step5_draft_is_accepted_as_it_is(tmp_path, monkeypatch):
    """The real draft format written by step 5 must pass untouched (nothing edited = valid)."""
    ids, labels = make_world(tmp_path)
    _, _, seen, vectors = s5.load_clusters(tmp_path / "clusters.csv", str(tmp_path / "emb"))
    cards = s5.build_cards(labels, seen, vectors)
    named = {c: {"name": f"Cluster {c}", "description": "d", "coherence": "clear", "reason": "r",
                 "size": cards[c]["size"]} for c in cards}
    topics = [{"name": f"Topic {i}", "description": "d", "cluster_ids": [2 * i, 2 * i + 1]}
              for i in range(3)]
    taxonomy = s5.build_taxonomy(topics, named, cards, noise_count=4)

    class Args: model = "m"; topic_model = None; think = False
    path = tmp_path / "draft.yaml"
    s5.write_yaml(str(path), taxonomy, Args())
    run_main(monkeypatch, tmp_path, str(path))
    frozen = yaml.safe_load(open(tmp_path / "taxonomy" / "taxonomy.v1.yaml", encoding="utf-8"))
    assert sum(t["post_count"] for t in frozen["topics"]) == 30 and frozen["other"]["post_count"] == 4


def test_subtopics_and_topics_are_listed_biggest_first():
    sizes = {0: 3, 1: 9, 2: 5, 3: 7}
    words = {c: [] for c in sizes}
    doc = {"topics": [{"name": "Small", "description": "d", "subtopics": [sub("A", 0), sub("B", 2)]},
                      {"name": "Big", "description": "d", "subtopics": [sub("C", 3), sub("D", 1)]}]}
    taxonomy, problems, _ = fin.finalize(doc, sizes, words, 0)
    assert problems == []
    assert [t["name"] for t in taxonomy["topics"]] == ["Big", "Small"]
    assert [s["name"] for s in taxonomy["topics"][0]["subtopics"]] == ["D", "C"]
