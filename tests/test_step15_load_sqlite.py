"""Tests for pipeline/step15_load_sqlite.py (Step 15): the SQLite loader.

All data here is made up (the real posts are private). The two promises that matter most:
  * every post keeps a link back to LinkedIn (post_url), and
  * the resource links found in a post survive the load and show up in the `library` view.
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import classifier as C  # noqa: E402
import classify_inputs as I  # noqa: E402
import review_rule as RV  # noqa: E402
import step15_load_sqlite as L  # noqa: E402

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "Retrieval-Augmented Generation", "description": "RAG systems.",
         "subtopics": [{"id": "rag.chunking", "name": "Chunking", "description": "Splitting documents."},
                       {"id": "rag.evaluation", "name": "RAG Evaluation", "description": "Measuring RAG."}]},
        {"id": "careers", "name": "Career Development", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
    ],
    "other": {"id": "other.general", "name": "Other / General", "description": "Fits nothing."},
}
RULE = {"version": "v1", "confidence_below": 0.78, "other_general": True, "topic_disagrees_with_clustering": True}

CSV_COLUMNS = ["activity_id", "saved_at", "post_url", "post_date_utc", "author", "text", "text_clean",
               "urls", "attachment_title", "attachment_site", "has_image", "source_file", "status"]


def url_of(aid: int) -> str:
    return f"https://www.linkedin.com/feed/update/urn:li:activity:{aid}"


def csv_row(aid, text="First line of the post\nsecond line", urls="", **over):
    row = {"activity_id": str(aid), "saved_at": "2026-10-30 10:00:00", "post_url": url_of(aid),
           "post_date_utc": "2026-10-29", "author": "Ada", "text": text, "text_clean": text, "urls": urls,
           "attachment_title": "", "attachment_site": "", "has_image": "False", "source_file": "sr1.html",
           "status": "captured"}
    row.update(over)
    return row


def record(aid, *, sub="rag.chunking", clustering="rag.chunking", self_conf=0.9, valid=1.0, partial=False,
           penalties=0, confidence=0.95, needs_review=False, key=None, run_stamp="2026-11-13T03:00:00+00:00",
           warnings=(), **over):
    rec = {
        "activity_id": aid, "topic_id": sub.split(".")[0] if sub != "other.general" else "other",
        "subtopic_id": sub, "difficulty": "intermediate", "content_type": "tutorial",
        "time": {"cover_minutes": 10, "understand_minutes": 20, "cover_bucket": "<15m", "understand_bucket": "15-30m",
                 "basis": "post_only", "stated_minutes": None},
        "alignment_score": 7.5, "alignment_band": "High", "alignment_subscores": {"applicability": 7},
        "summary": "A summary.", "key_points": ["a", "b"], "rationales": {"topic": "because"},
        "confidence": confidence,
        "confidence_parts": {"self": self_conf, "agree": 1.0, "valid": valid, "partial": partial, "penalties": penalties},
        "needs_review": needs_review, "review_reasons": ["low_confidence"] if needs_review else [],
        "review_rule_version": "v1", "review_rule_sha": "x" * 64,
        "warnings": list(warnings), "clustering_subtopic_id": clustering,
        "model": "gemma4:e4b", "prompt_version": "p1", "prompt_sha": "p" * 64, "taxonomy_version": "v1",
        "rubrics_version": "v2", "rubrics_sha": "r" * 64, "cache_key": key or f"key-{aid}-" + "0" * 40,
        "attempts": {"topic": 1, "rest": 1}, "retry_reasons": {"topic": [], "rest": []},
        "input_tokens": 100, "output_tokens": 50, "seconds": 10.0, "created_at": run_stamp,
    }
    rec.update(over)
    return rec


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def env(tmp_path):
    """A small world: 5 posts in the CSV (one without text), 4 classified, one with 3 resource links."""
    tax = tmp_path / "tax.yaml"
    tax.write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    rule = tmp_path / "rule.yaml"
    rule.write_text(yaml.safe_dump(RULE), encoding="utf-8")
    rows = [
        csv_row(1, urls="https://lnkd.in/aaa | https://example.com/b | https://lnkd.in/ccc"),
        csv_row(2, urls="https://lnkd.in/ddd", attachment_title="A Guide"),
        csv_row(3, urls=""),
        csv_row(4, urls=""),
        csv_row(5, text="", status="saved_after_export", saved_at=""),
    ]
    recs = [record(1), record(2, sub="careers.interviews", clustering="careers.interviews"),
            record(3, sub="rag.evaluation", clustering="other.general", self_conf=0.9, confidence=0.75,
                   needs_review=True),
            record(4, sub="rag.chunking", clustering="rag.evaluation", self_conf=0.6, confidence=0.9)]
    return {"dir": tmp_path, "tax": tax, "rule": rule, "rows": rows, "recs": recs,
            "csv": write_csv(tmp_path / "posts.csv", rows), "jsonl": write_jsonl(tmp_path / "run1.jsonl", recs),
            "db": tmp_path / "data" / "lib.sqlite"}


def do_load(env, **over):
    args = dict(db_path=env["db"], jsonl_path=env["jsonl"], csv_path=env["csv"], taxonomy_path=env["tax"],
                review_rule_path=env["rule"])
    args.update(over)
    return L.load(**args)


def connect(env):
    con = sqlite3.connect(env["db"])
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


@pytest.fixture(scope="module")
def tax_obj(tmp_path_factory):
    p = tmp_path_factory.mktemp("t") / "tax.yaml"
    p.write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    return I.load_taxonomy(p)


def a_rule(**over):
    base = dict(version="t", confidence_below=0.78, other_general=True, topic_disagrees=True, sha="s")
    base.update(over)
    return RV.ReviewRule(**base)


# ============================================================ links: the two promises

def test_every_item_keeps_its_post_link(env):
    do_load(env)
    with connect(env) as con:
        rows = con.execute("SELECT id, post_url FROM items").fetchall()
    assert len(rows) == 5
    assert all(r["post_url"] == url_of(int(r["id"])) for r in rows)


def test_every_library_row_has_a_post_link(env):
    do_load(env)
    with connect(env) as con:
        rows = con.execute("SELECT activity_id, post_url FROM library").fetchall()
    assert len(rows) == 4
    assert {r["activity_id"]: r["post_url"] for r in rows} == {i: url_of(i) for i in (1, 2, 3, 4)}


def test_resource_links_survive_the_load_in_order(env):
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM items WHERE id = '1'").fetchone()
    assert json.loads(row["resource_links"]) == ["https://lnkd.in/aaa", "https://example.com/b", "https://lnkd.in/ccc"]
    assert row["resource_link"] == "https://lnkd.in/aaa" and row["resource_link_count"] == 3


def test_a_post_without_resource_links_has_none(env):
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM items WHERE id = '3'").fetchone()
    assert row["resource_links"] == "[]" and row["resource_link"] is None and row["resource_link_count"] == 0


def test_the_library_view_exposes_both_links(env):
    do_load(env)
    with connect(env) as con:
        one = con.execute("SELECT * FROM library WHERE activity_id = 1").fetchone()
        none = con.execute("SELECT * FROM library WHERE activity_id = 3").fetchone()
    assert one["post_url"] == url_of(1) and one["resource_link"] == "https://lnkd.in/aaa"
    assert one["resource_link_count"] == 3 and len(json.loads(one["resource_links"])) == 3
    assert none["post_url"] == url_of(3) and none["resource_link"] is None


def test_the_number_of_posts_with_resource_links_matches_the_csv(env):
    rep = do_load(env)
    with connect(env) as con:
        n = con.execute("SELECT COUNT(*) FROM library WHERE resource_link_count > 0").fetchone()[0]
    assert n == 2 and rep.resource_link_posts == 2 and rep.problems == []


def test_a_row_without_a_post_link_stops_the_load_and_names_it(env):
    env["rows"][2]["post_url"] = ""
    write_csv(env["csv"], env["rows"])
    with pytest.raises(L.LoadError, match=r"activity_id 3 has no usable post_url"):
        do_load(env)
    assert not env["db"].exists()


def test_a_post_link_must_look_like_a_web_address(env):
    env["rows"][0]["post_url"] = "linkedin.com/feed/update/1"
    write_csv(env["csv"], env["rows"])
    with pytest.raises(L.LoadError, match="no usable post_url"):
        do_load(env)


def test_the_table_refuses_an_empty_post_link(env):
    do_load(env)
    with connect(env) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE items SET post_url = '' WHERE id = '1'")


def test_the_two_posts_cannot_share_a_link(env):
    do_load(env)
    with connect(env) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE items SET post_url = ? WHERE id = '2'", (url_of(1),))


# ============================================================ items and contents

def test_the_post_without_text_is_an_item_but_not_in_the_library(env):
    rep = do_load(env)
    with connect(env) as con:
        item = con.execute("SELECT * FROM items WHERE id = '5'").fetchone()
        has_content = con.execute("SELECT COUNT(*) FROM contents WHERE item_id = '5'").fetchone()[0]
        in_library = con.execute("SELECT COUNT(*) FROM library WHERE id = '5'").fetchone()[0]
    assert item["status"] == "no_text" and item["post_url"] == url_of(5) and item["saved_at"] is None
    assert item["capture_status"] == "saved_after_export"
    assert has_content == 0 and in_library == 0 and rep.items_without_text == 1 and rep.items == 5


def test_item_fields(env):
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM items WHERE id = '1'").fetchone()
    assert row["activity_id"] == 1 and row["author"] == "Ada" and row["kind"] == "post"
    assert row["collector"] == "sr1.html" and row["post_date"] == "2026-10-29"
    assert row["saved_at"] == "2026-10-30 10:00:00" and row["status"] == "classified"
    assert row["capture_status"] == "captured" and row["duplicate_of"] is None


def test_the_title_is_the_attachment_title_or_the_first_line():
    assert L.make_title("A Guide", "text") == "A Guide"
    assert L.make_title("", "\n  \nFirst line\nsecond") == "First line"
    assert L.make_title("  ", "") == ""
    assert L.make_title("", "x" * 200) == "x" * (L.TITLE_CHARS - 1) + "…"
    assert L.make_title("", "x" * L.TITLE_CHARS) == "x" * L.TITLE_CHARS
    assert L.make_title("y" * 200, "text") == "y" * L.TITLE_CHARS


def test_titles_are_stored(env):
    do_load(env)
    with connect(env) as con:
        titles = {r["id"]: r["title"] for r in con.execute("SELECT id, title FROM items")}
    assert titles["1"] == "First line of the post" and titles["2"] == "A Guide" and titles["5"] is None


def test_split_links():
    assert L.split_links("") == [] and L.split_links(None) == []
    assert L.split_links("https://a | https://b") == ["https://a", "https://b"]
    assert L.split_links(" https://a |  | https://b ") == ["https://a", "https://b"]
    assert L.split_links("https://a") == ["https://a"]


def test_contents(env):
    env["rows"][0]["has_image"] = "True"
    env["rows"][0]["attachment_site"] = "lnkd.in"
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM contents WHERE item_id = '1'").fetchone()
    assert row["body_text"] == "First line of the post\nsecond line" and len(row["content_hash"]) == 64
    m = json.loads(row["metrics"])
    assert m["word_count"] == 7 and m["char_count"] == 34 and m["has_image"] is True
    assert m["url_count"] == 3 and m["attachment_site"] == "lnkd.in"
    assert row["partial"] == 0


def test_the_content_hash_follows_the_text(env):
    do_load(env)
    with connect(env) as con:
        h = {r["item_id"]: r["content_hash"] for r in con.execute("SELECT item_id, content_hash FROM contents")}
    assert len(set(h.values())) == 1 or h["1"] == h["3"]            # same text -> same hash
    env["rows"][2]["text"] = env["rows"][2]["text_clean"] = "different text"
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        h2 = {r["item_id"]: r["content_hash"] for r in con.execute("SELECT item_id, content_hash FROM contents")}
    assert h2["3"] != h["3"] and h2["1"] == h["1"]
    with connect(env) as con:
        assert con.execute("SELECT body_text FROM contents WHERE item_id = '3'").fetchone()[0] == "different text"


def test_a_cut_off_post_is_marked_partial(env):
    env["recs"][0] = record(1, warnings=["text_partial"])
    write_jsonl(env["jsonl"], env["recs"])
    do_load(env)
    with connect(env) as con:
        flags = {r["item_id"]: r["partial"] for r in con.execute("SELECT item_id, partial FROM contents")}
    assert flags["1"] == 1 and flags["2"] == 0


def test_text_falls_back_to_the_raw_text_column(env):
    env["rows"][3]["text_clean"] = ""
    env["rows"][3]["text"] = "raw only"
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        assert con.execute("SELECT body_text FROM contents WHERE item_id = '4'").fetchone()[0] == "raw only"


# ============================================================ the review rule at load time

def test_a_post_the_clustering_could_not_place_has_no_agreement_part(tax_obj):
    rec = record(3, sub="rag.evaluation", clustering="other.general", self_conf=0.9, valid=1.0)
    new = L.rescore(rec, tax_obj, a_rule())
    assert new["confidence"] == 0.93                       # (0.45*0.9 + 0.20*1.0) / 0.65
    assert new["confidence_parts"]["agree"] is None
    assert new["needs_review"] is False and new["review_reasons"] == []


def test_a_post_without_a_clustering_opinion_is_scored_the_same_way(tax_obj):
    rec = record(3, clustering=None, self_conf=0.9)
    assert L.rescore(rec, tax_obj, a_rule())["confidence"] == 0.93


def test_the_model_choosing_other_general_still_disagrees_with_a_real_clustering(tax_obj):
    rec = record(3, sub="other.general", clustering="rag.chunking", self_conf=0.9)
    new = L.rescore(rec, tax_obj, a_rule())
    assert new["confidence_parts"]["agree"] == 0.4 and new["confidence"] == 0.75   # 0.405 + 0.14 + 0.2
    assert new["review_reasons"] == ["low_confidence", "other_general", "topic_disagrees_with_clustering"]


def test_agreement_levels_are_unchanged_for_posts_the_clustering_placed(tax_obj):
    same = L.rescore(record(1, clustering="rag.chunking", self_conf=0.6), tax_obj, a_rule())
    topic = L.rescore(record(1, clustering="rag.evaluation", self_conf=0.6), tax_obj, a_rule())
    other = L.rescore(record(1, clustering="careers.interviews", self_conf=0.6), tax_obj, a_rule())
    assert (same["confidence"], topic["confidence"], other["confidence"]) == (0.82, 0.72, 0.61)
    assert same["confidence_parts"]["agree"] == 1.0 and topic["confidence_parts"]["agree"] == 0.7


def test_rescoring_reproduces_what_the_classifier_wrote_for_a_placed_post(tax_obj):
    conf, parts = C.compute_confidence(self_confidence=0.9, agree=C.agreement("rag.chunking", "rag.evaluation", tax_obj),
                                       first_try=False, partial=True, penalties=1)
    rec = record(1, clustering="rag.evaluation", self_conf=0.9, valid=parts["valid"], partial=True,
                 penalties=1, confidence=conf)
    new = L.rescore(rec, tax_obj, a_rule())
    assert new["confidence"] == conf and new["confidence_parts"] == parts


def test_a_retry_lowers_the_valid_part_when_rescoring(tax_obj):
    first = L.rescore(record(1, valid=1.0), tax_obj, a_rule())["confidence"]
    retry = L.rescore(record(1, valid=0.7), tax_obj, a_rule())
    assert retry["confidence_parts"]["valid"] == 0.7 and retry["confidence"] < first


def test_partial_and_penalties_are_used_when_rescoring(tax_obj):
    plain = L.rescore(record(1), tax_obj, a_rule())["confidence"]
    assert L.rescore(record(1, partial=True), tax_obj, a_rule())["confidence"] < plain
    assert L.rescore(record(1, penalties=2), tax_obj, a_rule())["confidence"] == round(plain - 0.2, 2)


def test_a_topic_disagreement_is_flagged_even_when_the_confidence_is_high(tax_obj):
    rec = record(1, sub="careers.interviews", clustering="rag.chunking", self_conf=1.0)
    new = L.rescore(rec, tax_obj, a_rule(confidence_below=0.0))
    assert new["review_reasons"] == ["topic_disagrees_with_clustering"] and new["needs_review"] is True


def test_a_new_threshold_changes_the_flags_without_a_new_run(env):
    do_load(env)
    with connect(env) as con:
        before = con.execute("SELECT COUNT(*) FROM library WHERE needs_review = 1").fetchone()[0]
    env["rule"].write_text(yaml.safe_dump({**RULE, "confidence_below": 0.99}), encoding="utf-8")
    rep = do_load(env)
    with connect(env) as con:
        after = con.execute("SELECT COUNT(*) FROM library WHERE needs_review = 1").fetchone()[0]
        sha = con.execute("SELECT DISTINCT review_rule_sha FROM classifications").fetchall()
    assert after > before and rep.inserted == 0 and rep.refreshed == 4 and len(sha) == 1


def test_the_loaded_values_are_the_recomputed_ones_and_the_run_values_are_kept(env):
    rep = do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM classifications WHERE item_id = '3'").fetchone()
    assert row["confidence"] == 0.93 and row["needs_review"] == 0 and row["review_reasons"] == "[]"
    assert row["confidence_as_run"] == 0.75 and row["needs_review_as_run"] == 1
    assert row["review_rule_version"] == "v1" and row["review_rule_sha"] != "x" * 64
    assert (rep.needs_review_as_run, rep.needs_review_now) == (1, 1)
    assert (rep.noise_posts, rep.noise_flagged_as_run, rep.noise_flagged_now) == (1, 1, 0)


def test_review_reasons_are_stored_as_a_list(env):
    env["recs"][3] = record(4, sub="careers.interviews", clustering="rag.chunking", self_conf=0.9)
    write_jsonl(env["jsonl"], env["recs"])
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM library WHERE activity_id = 4").fetchone()
    assert json.loads(row["review_reasons"]) == ["low_confidence", "topic_disagrees_with_clustering"]
    assert row["needs_review"] == 1


# ============================================================ classifications and the library view

def test_classification_fields(env):
    do_load(env)
    with connect(env) as con:
        r = con.execute("SELECT * FROM classifications WHERE item_id = '2'").fetchone()
    assert r["topic_id"] == "careers" and r["subtopic_id"] == "careers.interviews"
    assert r["difficulty"] == "intermediate" and r["content_type"] == "tutorial"
    assert (r["cover_minutes"], r["understand_minutes"]) == (10, 20)
    assert (r["cover_bucket"], r["understand_bucket"], r["time_basis"]) == ("<15m", "15-30m", "post_only")
    assert json.loads(r["time_details"])["stated_minutes"] is None
    assert r["alignment_score"] == 7.5 and r["alignment_band"] == "High"
    assert json.loads(r["alignment_subscores"]) == {"applicability": 7}
    assert r["summary"] == "A summary." and json.loads(r["key_points"]) == ["a", "b"]
    assert json.loads(r["rationales"]) == {"topic": "because"}
    assert json.loads(r["attempts"]) == {"topic": 1, "rest": 1}
    assert r["model"] == "gemma4:e4b" and r["prompt_version"] == "p1" and r["rubrics_version"] == "v2"
    assert r["taxonomy_version"] == "v1" and r["cache_key"] == "key-2-" + "0" * 40
    assert (r["input_tokens"], r["output_tokens"], r["seconds"]) == (100, 50, 10.0)
    assert r["clustering_subtopic_id"] == "careers.interviews" and r["is_active"] == 1
    assert r["id"] == "cls_" + r["cache_key"][:16] and r["created_at"] == "2026-11-13T03:00:00+00:00"


def test_the_library_view_has_names_and_labels(env):
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM library WHERE activity_id = 2").fetchone()
        other = con.execute("""SELECT t.name, s.name FROM topics t JOIN subtopics s
                               ON s.taxonomy_version = t.taxonomy_version AND s.topic_id = t.id
                               WHERE s.id = 'other.general'""").fetchone()
    assert row["topic_name"] == "Career Development" and row["subtopic_name"] == "Interviews"
    assert row["title"] == "A Guide" and row["author"] == "Ada" and row["model"] == "gemma4:e4b"
    assert row["difficulty"] == "intermediate" and row["content_type"] == "tutorial"
    assert (row["cover_minutes"], row["understand_minutes"]) == (10, 20)
    assert row["alignment_score"] == 7.5 and row["summary"] == "A summary." and row["time_basis"] == "post_only"
    assert tuple(other) == ("Other / General", "Other / General")


def test_the_library_shows_other_general_with_its_name(env):
    env["recs"][3] = record(4, sub="other.general", clustering="rag.chunking")
    write_jsonl(env["jsonl"], env["recs"])
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT topic_name, subtopic_name FROM library WHERE activity_id = 4").fetchone()
    assert tuple(row) == ("Other / General", "Other / General")


def test_a_duplicate_is_left_out_of_the_library(env):
    do_load(env)
    with connect(env) as con:
        con.execute("UPDATE items SET duplicate_of = '1' WHERE id = '2'")
        ids = {r[0] for r in con.execute("SELECT activity_id FROM library")}
    assert ids == {1, 3, 4}


def test_the_taxonomy_is_stored(env):
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT * FROM taxonomy_versions").fetchone()
        n_topics = con.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        n_subs = con.execute("SELECT COUNT(*) FROM subtopics").fetchone()[0]
    assert row["version"] == "v1" and len(row["sha256"]) == 64
    assert json.loads(row["tree"])["topics"][0]["id"] == "rag"
    assert (n_topics, n_subs) == (3, 4)               # rag, careers, other / 3 subtopics + other.general


def test_the_run_row(env):
    rep = do_load(env)
    with connect(env) as con:
        run = con.execute("SELECT * FROM runs").fetchone()
    snap = json.loads(run["config_snapshot"])
    assert rep.run_id == "run_run1" and run["id"] == "run_run1"
    assert (run["items_ok"], run["items_failed"], run["input_tokens"], run["output_tokens"]) == (4, 0, 400, 200)
    assert run["started_at"] == run["finished_at"] == "2026-11-13T03:00:00+00:00"
    assert snap["model"] == "gemma4:e4b" and snap["prompt_sha"] == "p" * 64 and snap["rubrics_sha"] == "r" * 64
    assert snap["file"] == "run1.jsonl" and len(snap["file_sha256"]) == 64 and snap["taxonomy_version"] == "v1"
    assert snap["review_rule_as_run"]["sha"] == "x" * 64 and snap["review_rule_at_load"]["confidence_below"] == 0.78
    assert "run1.jsonl" in run["command"]


def test_the_run_times_span_the_records(env):
    env["recs"][0] = record(1, run_stamp="2026-11-13T05:00:00+00:00")
    env["recs"][1] = record(2, sub="careers.interviews", clustering="careers.interviews", run_stamp="2026-11-13T01:00:00+00:00")
    write_jsonl(env["jsonl"], env["recs"])
    do_load(env)
    with connect(env) as con:
        run = con.execute("SELECT started_at, finished_at FROM runs").fetchone()
    assert (run["started_at"], run["finished_at"]) == ("2026-11-13T01:00:00+00:00", "2026-11-13T05:00:00+00:00")


def test_each_classification_points_at_its_run(env):
    do_load(env)
    with connect(env) as con:
        runs = {r[0] for r in con.execute("SELECT run_id FROM classifications")}
    assert runs == {"run_run1"}


# ============================================================ loading again, history, reviews

def test_loading_the_same_file_twice_changes_nothing_but_the_refresh(env):
    first = do_load(env)
    second = do_load(env)
    with connect(env) as con:
        n_cls = con.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        n_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        n_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert (first.inserted, first.refreshed) == (4, 0) and (second.inserted, second.refreshed) == (0, 4)
    assert (n_cls, n_items, n_runs) == (4, 5, 1) and second.superseded == 0


def test_a_newer_run_becomes_active_and_the_older_row_is_kept(env):
    do_load(env)
    newer = [record(1, sub="rag.evaluation", clustering="rag.evaluation", key="newer-1" + "0" * 40,
                    run_stamp="2026-11-14T03:00:00+00:00")]
    write_jsonl(env["dir"] / "run2.jsonl", newer)
    rep = do_load(env, jsonl_path=env["dir"] / "run2.jsonl")
    with connect(env) as con:
        rows = con.execute("SELECT subtopic_id, is_active, run_id FROM classifications WHERE item_id = '1' ORDER BY created_at").fetchall()
        lib = con.execute("SELECT subtopic_id FROM library WHERE activity_id = 1").fetchall()
    assert [tuple(r) for r in rows] == [("rag.chunking", 0, "run_run1"), ("rag.evaluation", 1, "run_run2")]
    assert [tuple(r) for r in lib] == [("rag.evaluation",)]
    assert (rep.inserted, rep.superseded) == (1, 1)


def test_posts_missing_from_a_newer_run_keep_their_active_row(env):
    do_load(env)
    write_jsonl(env["dir"] / "run2.jsonl", [record(1, key="newer-1" + "0" * 40)])
    do_load(env, jsonl_path=env["dir"] / "run2.jsonl")
    with connect(env) as con:
        n = con.execute("SELECT COUNT(*) FROM library").fetchone()[0]
    assert n == 4


def test_loading_an_older_file_again_does_not_take_the_active_row_back(env):
    do_load(env)
    write_jsonl(env["dir"] / "run2.jsonl", [record(1, sub="rag.evaluation", clustering="rag.evaluation", key="newer-1" + "0" * 40)])
    do_load(env, jsonl_path=env["dir"] / "run2.jsonl")
    do_load(env)                                            # run1 again
    with connect(env) as con:
        active = con.execute("SELECT run_id FROM classifications WHERE item_id = '1' AND is_active = 1").fetchall()
    assert [r[0] for r in active] == ["run_run2"]


def test_the_last_line_of_a_post_wins_inside_one_file(env):
    first = record(1, sub="rag.chunking", key="first-1" + "0" * 40)
    last = record(1, sub="rag.evaluation", clustering="rag.evaluation", key="last-1" + "0" * 40)
    write_jsonl(env["jsonl"], [first, last] + env["recs"][1:])
    rep = do_load(env)
    with connect(env) as con:
        rows = con.execute("SELECT subtopic_id FROM classifications WHERE item_id = '1'").fetchall()
    assert [r[0] for r in rows] == ["rag.evaluation"] and rep.duplicate_lines == 1 and rep.records == 4


def test_failure_lines_in_a_results_file_are_ignored(env):
    write_jsonl(env["jsonl"], env["recs"] + [{"activity_id": 5, "error": "timeout: x", "created_at": "t"}])
    rep = do_load(env)
    assert rep.records == 4 and rep.problems == []


def test_your_reviews_survive_loading_again(env):
    do_load(env)
    with connect(env) as con:
        cid = con.execute("SELECT id FROM classifications WHERE item_id = '1'").fetchone()[0]
        con.execute("INSERT INTO reviews(id, item_id, classification_id, corrected_fields, note, created_at) "
                    "VALUES ('r1', '1', ?, '{\"difficulty\": \"advanced\"}', 'n', 't')", (cid,))
        con.commit()
    do_load(env)
    write_jsonl(env["dir"] / "run2.jsonl", [record(1, key="newer-1" + "0" * 40)])
    do_load(env, jsonl_path=env["dir"] / "run2.jsonl")
    with connect(env) as con:
        row = con.execute("SELECT * FROM reviews").fetchone()
    assert row["id"] == "r1" and row["classification_id"] == cid and row["use_as_example"] == 1


def test_a_review_row_needs_an_existing_item(env):
    do_load(env)
    with connect(env) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO reviews(id, item_id, corrected_fields, created_at) VALUES ('r', 'nope', '{}', 't')")


def test_only_one_active_row_per_item_is_allowed(env):
    do_load(env)
    with connect(env) as con:
        con.execute("UPDATE classifications SET is_active = 0 WHERE item_id = '1'")
        con.execute("""INSERT INTO classifications SELECT 'dup', item_id, topic_id, subtopic_id, difficulty,
            content_type, cover_minutes, understand_minutes, cover_bucket, understand_bucket, time_basis,
            time_details, alignment_score, alignment_band, alignment_subscores, summary, key_points,
            rationales, confidence, confidence_parts, needs_review, review_reasons, review_rule_version,
            review_rule_sha, confidence_as_run, needs_review_as_run, clustering_subtopic_id, warnings,
            attempts, retry_reasons, 1, model, prompt_version, prompt_sha, rubrics_version, rubrics_sha,
            taxonomy_version, 'other-key', run_id, input_tokens, output_tokens, seconds, created_at
            FROM classifications WHERE item_id = '1'""")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE classifications SET is_active = 1 WHERE item_id = '1'")


def test_the_same_cache_key_cannot_be_stored_twice(env):
    do_load(env)
    with connect(env) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE classifications SET cache_key = (SELECT cache_key FROM classifications WHERE item_id = '2') WHERE item_id = '1'")


# ============================================================ bad input: everything is reported, nothing is saved

def test_a_result_for_a_post_that_is_not_in_the_csv(env):
    write_jsonl(env["jsonl"], env["recs"] + [record(99)])
    with pytest.raises(L.LoadError, match="activity_id 99 is in the results but not in the posts CSV"):
        do_load(env)


def test_results_made_with_another_taxonomy_version_are_refused(env):
    write_jsonl(env["jsonl"], [record(1, taxonomy_version="v2")])
    with pytest.raises(L.LoadError, match="classified with taxonomy 'v2' but the taxonomy file is 'v1'"):
        do_load(env)


def test_a_subtopic_that_is_not_in_the_taxonomy_is_refused(env):
    write_jsonl(env["jsonl"], [record(1, sub="rag.unknown")])
    with pytest.raises(L.LoadError, match="subtopic 'rag.unknown' is not in the taxonomy"):
        do_load(env)


def test_all_problems_are_listed_together(env):
    write_jsonl(env["jsonl"], [record(98), record(99, taxonomy_version="v2")])
    with pytest.raises(L.LoadError) as err:
        do_load(env)
    text = str(err.value)
    assert "activity_id 98" in text and "activity_id 99" in text and "taxonomy 'v2'" in text


def test_a_long_list_of_problems_is_cut(env):
    write_jsonl(env["jsonl"], [record(1000 + i) for i in range(60)])
    with pytest.raises(L.LoadError, match=r"and 10 more") as err:
        do_load(env)
    assert len(str(err.value).splitlines()) == 1 + 50 + 1       # the heading, 50 problems, the "more" line


def test_a_line_that_is_not_json(env):
    env["jsonl"].write_text(json.dumps(record(1)) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match="line 2: not valid JSON"):
        do_load(env)


def test_a_line_without_an_activity_id(env):
    env["jsonl"].write_text(json.dumps({"summary": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match="line 1: no whole-number activity_id"):
        do_load(env)


def test_a_record_with_missing_fields(env):
    rec = record(1)
    del rec["confidence_parts"], rec["cache_key"]
    write_jsonl(env["jsonl"], [rec])
    with pytest.raises(L.LoadError, match=r"lacks fields \['confidence_parts', 'cache_key'\]"):
        do_load(env)


def test_an_empty_results_file(env):
    env["jsonl"].write_text("\n\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match="has no classified posts"):
        do_load(env)


def test_a_missing_results_file(env):
    with pytest.raises(L.LoadError, match="cannot read"):
        do_load(env, jsonl_path=env["dir"] / "nope.jsonl")


def test_a_csv_without_the_needed_columns(env):
    env["csv"].write_text("activity_id,text\n1,hello\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match=r"missing columns \['post_url'\]"):
        do_load(env)


def test_a_missing_csv(env):
    with pytest.raises(L.LoadError, match="cannot read"):
        do_load(env, csv_path=env["dir"] / "nope.csv")


def test_a_csv_with_a_repeated_or_bad_activity_id(env):
    env["rows"].append(csv_row(1))
    env["rows"].append(csv_row("abc"))
    write_csv(env["csv"], env["rows"])
    with pytest.raises(L.LoadError) as err:
        do_load(env)
    assert "activity_id 1 appears twice" in str(err.value) and "'abc' is not a whole number" in str(err.value)


def test_a_bad_taxonomy_or_rule_file(env):
    env["tax"].write_text("version: v1\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match="taxonomy problems"):
        do_load(env)
    env["tax"].write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    env["rule"].write_text("version: v1\n", encoding="utf-8")
    with pytest.raises(L.LoadError, match="review rule problems"):
        do_load(env)


def test_a_changed_taxonomy_with_the_same_version_name_is_refused(env):
    do_load(env)
    changed = {**TAXONOMY, "other": {**TAXONOMY["other"], "description": "Changed text."}}
    env["tax"].write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(L.LoadError, match="already in the database with different content"):
        do_load(env)


def test_a_value_the_table_rules_refuse_saves_nothing(env):
    bad = record(2, sub="careers.interviews", clustering="careers.interviews", difficulty="godlike")
    write_jsonl(env["jsonl"], [record(1), bad])
    with pytest.raises(L.LoadError, match=r"database problems \(nothing was saved\)"):
        do_load(env)
    with connect(env) as con:
        assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM classifications").fetchone()[0] == 0


def test_understand_time_cannot_be_shorter_than_cover_time(env):
    rec = record(1)
    rec["time"] = {**rec["time"], "cover_minutes": 30, "understand_minutes": 10}
    write_jsonl(env["jsonl"], [rec])
    with pytest.raises(L.LoadError, match="nothing was saved"):
        do_load(env)


def test_the_alignment_score_must_be_between_0_and_10(env):
    write_jsonl(env["jsonl"], [record(1, alignment_score=11)])
    with pytest.raises(L.LoadError, match="nothing was saved"):
        do_load(env)


def test_a_file_that_is_not_one_of_ours_is_refused(env):
    env["db"].parent.mkdir(parents=True)
    con = sqlite3.connect(env["db"])
    con.execute("CREATE TABLE other(x)")
    con.execute("PRAGMA user_version = 7")
    con.commit()
    con.close()
    with pytest.raises(L.LoadError, match="not a database made by this loader"):
        do_load(env)


def test_an_empty_database_file_is_accepted_and_filled(env):
    env["db"].parent.mkdir(parents=True)
    env["db"].write_bytes(b"")
    assert do_load(env).inserted == 4


def test_the_database_folder_is_created(env):
    assert not env["db"].parent.exists()
    do_load(env)
    assert env["db"].exists()


def test_foreign_keys_are_enforced(env):
    do_load(env)
    con = L.open_db(env["db"])
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM taxonomy_versions")
    finally:
        con.close()


def test_the_schema_version_is_stored(env):
    do_load(env)
    with connect(env) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == L.SCHEMA_VERSION == 1


def test_a_results_file_with_a_byte_order_mark_loads(env):
    env["jsonl"].write_bytes(b"\xef\xbb\xbf" + (json.dumps(record(1)) + "\n").encode("utf-8"))
    assert do_load(env).records == 1


def test_the_post_link_is_stored_without_stray_spaces(env):
    env["rows"][0]["post_url"] = "  " + url_of(1) + " "
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        assert con.execute("SELECT post_url FROM items WHERE id = '1'").fetchone()[0] == url_of(1)


def test_missing_source_file_and_status_get_plain_defaults(env):
    env["rows"][0]["source_file"] = ""
    env["rows"][0]["status"] = ""
    env["rows"][0]["author"] = ""
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        row = con.execute("SELECT collector, capture_status, author FROM items WHERE id = '1'").fetchone()
    assert tuple(row) == ("unknown", "captured", None)


def test_a_post_whose_text_disappears_loses_its_content_row(env):
    do_load(env)
    env["rows"][3]["text"] = env["rows"][3]["text_clean"] = ""
    write_csv(env["csv"], env["rows"])
    write_jsonl(env["jsonl"], env["recs"][:3])
    do_load(env)
    with connect(env) as con:
        assert con.execute("SELECT COUNT(*) FROM contents WHERE item_id = '4'").fetchone()[0] == 0
        assert con.execute("SELECT status FROM items WHERE id = '4'").fetchone()[0] == "no_text"


def test_the_run_row_follows_the_file_when_it_grows(env):
    write_jsonl(env["jsonl"], env["recs"][:2])
    do_load(env)
    write_jsonl(env["jsonl"], env["recs"])
    do_load(env)
    with connect(env) as con:
        run = con.execute("SELECT * FROM runs").fetchone()
        n = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 1 and run["items_ok"] == 4 and run["input_tokens"] == 400
    assert len(json.loads(run["config_snapshot"])["file_sha256"]) == 64


def test_cover_time_must_be_above_zero(env):
    rec = record(1)
    rec["time"] = {**rec["time"], "cover_minutes": 0, "understand_minutes": 0}
    write_jsonl(env["jsonl"], [rec])
    with pytest.raises(L.LoadError, match="nothing was saved"):
        do_load(env)


def test_deleting_an_item_removes_its_text_and_labels(env):
    do_load(env)
    with connect(env) as con:
        con.execute("DELETE FROM items WHERE id = '1'")
        assert con.execute("SELECT COUNT(*) FROM contents WHERE item_id = '1'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM classifications WHERE item_id = '1'").fetchone()[0] == 0


def test_a_changed_post_link_is_updated_on_reload(env):
    do_load(env)
    env["rows"][0]["post_url"] = "https://www.linkedin.com/posts/new-address-1"
    write_csv(env["csv"], env["rows"])
    do_load(env)
    with connect(env) as con:
        assert con.execute("SELECT post_url FROM library WHERE activity_id = 1").fetchone()[0] == "https://www.linkedin.com/posts/new-address-1"


def test_names_come_from_the_taxonomy_version_each_row_was_made_with(env):
    do_load(env)
    v2 = {**TAXONOMY, "version": "v2"}
    v2["topics"] = [{**TAXONOMY["topics"][0], "name": "RAG, renamed"}, TAXONOMY["topics"][1]]
    tax2 = env["dir"] / "tax2.yaml"
    tax2.write_text(yaml.safe_dump(v2), encoding="utf-8")
    write_jsonl(env["dir"] / "run2.jsonl", [record(1, key="v2-1" + "0" * 40, taxonomy_version="v2")])
    do_load(env, jsonl_path=env["dir"] / "run2.jsonl", taxonomy_path=tax2)
    with connect(env) as con:
        rows = con.execute("SELECT activity_id, topic_name FROM library ORDER BY activity_id").fetchall()
    assert [tuple(r) for r in rows] == [(1, "RAG, renamed"), (2, "Career Development"),
                                        (3, "Retrieval-Augmented Generation"), (4, "Retrieval-Augmented Generation")]


def test_the_load_checks_the_database_against_the_csv(env, monkeypatch):
    seen = []
    real = L.verify
    monkeypatch.setattr(L, "verify", lambda con, posts=None: seen.append(posts) or real(con, posts))
    do_load(env)
    assert len(seen) == 1 and seen[0] is not None and set(seen[0]) == {1, 2, 3, 4, 5}


# ============================================================ the checks after a load

def test_verify_passes_on_a_good_load_and_reports_nothing(env):
    assert do_load(env).problems == []


def test_verify_notices_a_classified_item_without_an_active_row(env):
    do_load(env)
    with connect(env) as con:
        con.execute("UPDATE classifications SET is_active = 0 WHERE item_id = '1'")
        assert any("no active classification" in p for p in L.verify(con))


def test_verify_notices_lost_resource_links(env):
    do_load(env)
    posts = L.read_posts_csv(env["csv"])
    with connect(env) as con:
        con.execute("UPDATE items SET resource_links = '[]', resource_link = NULL, resource_link_count = 0 WHERE id = '1'")
        assert any("resource links" in p for p in L.verify(con, posts))


def test_verify_notices_a_count_without_links(env):
    do_load(env)
    with connect(env) as con:
        con.execute("UPDATE items SET resource_link = NULL WHERE id = '1'")
        assert any("count resource links but have none" in p for p in L.verify(con))


def test_verify_notices_a_dangling_pointer(env):
    do_load(env)
    with connect(env) as con:
        con.execute("PRAGMA foreign_keys = OFF")
        con.execute("INSERT INTO reviews(id, item_id, corrected_fields, created_at) VALUES ('r', 'nope', '{}', 't')")
        assert any("point at something that does not exist" in p for p in L.verify(con))


def test_verify_notices_a_missing_post_link_in_the_library(env):
    do_load(env)
    with connect(env) as con:
        con.execute("PRAGMA ignore_check_constraints = ON")
        con.execute("UPDATE items SET post_url = '' WHERE id = '2'")
        problems = L.verify(con)
    assert any("items have no post link" in p for p in problems) and any("library rows have no post link" in p for p in problems)


def test_verify_notices_two_active_rows(env):
    do_load(env)
    with connect(env) as con:
        con.execute("DROP INDEX ux_active_cls")
        con.execute("""INSERT INTO classifications SELECT 'dup', item_id, topic_id, subtopic_id, difficulty,
            content_type, cover_minutes, understand_minutes, cover_bucket, understand_bucket, time_basis,
            time_details, alignment_score, alignment_band, alignment_subscores, summary, key_points,
            rationales, confidence, confidence_parts, needs_review, review_reasons, review_rule_version,
            review_rule_sha, confidence_as_run, needs_review_as_run, clustering_subtopic_id, warnings,
            attempts, retry_reasons, 1, model, prompt_version, prompt_sha, rubrics_version, rubrics_sha,
            taxonomy_version, 'other-key', run_id, input_tokens, output_tokens, seconds, created_at
            FROM classifications WHERE item_id = '1'""")
        assert any("more than one active classification" in p for p in L.verify(con))


# ============================================================ command line

def cli(env, *extra):
    return L.main(["--jsonl", str(env["jsonl"]), "--csv", str(env["csv"]), "--taxonomy", str(env["tax"]),
                   "--db", str(env["db"]), "--review-rule", str(env["rule"]), *extra])


def test_cli_success(env, capsys):
    assert cli(env) == 0
    out = capsys.readouterr().out
    assert "items: 5 (1 without text" in out and "4 new, 0 refreshed" in out
    assert "posts with resource links: 2" in out and "needs review: 1 of 4 now, 1 as the run wrote it" in out
    assert "clustering-noise posts: 1; flagged now 0, as the run wrote it 1" in out and "checks passed" in out


def test_cli_reports_repeated_lines(env, capsys):
    write_jsonl(env["jsonl"], env["recs"] + [record(1, key="other-1" + "0" * 40)])
    assert cli(env) == 0
    assert "1 repeated lines" in capsys.readouterr().out


def test_cli_bad_input_exits_2_and_says_why(env, capsys):
    write_jsonl(env["jsonl"], [record(99)])
    assert cli(env) == 2
    assert "CANNOT LOAD" in capsys.readouterr().out and not env["db"].exists()


def test_cli_exits_1_when_a_check_fails(env, capsys, monkeypatch):
    monkeypatch.setattr(L, "verify", lambda con, posts=None: ["something is wrong"])
    assert cli(env) == 1
    out = capsys.readouterr().out
    assert "CHECKS FAILED" in out and "something is wrong" in out


def test_cli_default_database_path():
    assert L.build_parser().parse_args(["--jsonl", "a", "--csv", "b", "--taxonomy", "c"]).db == "data/library.sqlite"


def test_cli_default_review_rule_is_the_shipped_file():
    args = L.build_parser().parse_args(["--jsonl", "a", "--csv", "b", "--taxonomy", "c"])
    assert Path(args.review_rule) == RV.DEFAULT_PATH


def test_the_shipped_review_rule_loads_with_the_loader(env):
    rep = do_load(env, review_rule_path=RV.DEFAULT_PATH)
    assert rep.problems == [] and rep.inserted == 4
