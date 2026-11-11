"""Tests for pipeline/step10_compare.py (Step 10c).

A tiny world of four posts is used everywhere, small enough to check every number by hand:

  gold  label sub / type / difficulty        classifier says              clustering says
  g01   a.x / tutorial / beginner            a.x / tutorial / beginner    a.x          (all right)
  g02   a.y / news / intermediate            a.z / miscellaneous / advanced   a.y      (topic right, sub wrong)
  g03   b.x / (blank) / advanced             c.x / tutorial / beginner    c.x          (all wrong; difficulty two lower)
  g04   b.y / miscellaneous / (blank)        b.y / miscellaneous / expert b.z          (all right but clustering sub)
"""
import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import step10_compare as X  # noqa: E402

CP = X.CompareError

PLAIN_ROWS = [
    ("g01", "1001", "a.x", "tutorial", "beginner"),
    ("g02", "1002", "a.y", "news", "intermediate"),
    ("g03", "1003", "b.x", "", "advanced"),
    ("g04", "1004", "b.y", "miscellaneous", ""),
]
# reviewed file: the original labels differ from the reviewed ones in g02's type and g03's difficulty
REVIEWED_ROWS = [
    ("g01", "1001", "a.x", "a.x", "tutorial", "tutorial", "beginner", "beginner"),
    ("g02", "1002", "a.y", "a.y", "tutorial", "news", "intermediate", "intermediate"),
    ("g03", "1003", "b.x", "b.x", "", "", "intermediate", "advanced"),
    ("g04", "1004", "b.y", "b.y", "miscellaneous", "miscellaneous", "", ""),
]
KEY_ROWS = [("g01", "1001", "random", "1", "a.x"), ("g02", "1002", "random", "2", "a.y"),
            ("g03", "1003", "stated", "3", "c.x"), ("g04", "1004", "generic", "4", "b.z")]

RUN = {
    "1001": dict(sub="a.x", ct="tutorial", df="beginner", conf=0.9, att=(1, 1), basis="post_only", sec=10.0, nr=False),
    "1002": dict(sub="a.z", ct="miscellaneous", df="advanced", conf=0.6, att=(1, 2), basis="stated_in_post", sec=20.0, nr=True),
    "1003": dict(sub="c.x", ct="tutorial", df="beginner", conf=0.5, att=(2, 1), basis="post_only", sec=30.0, nr=False),
    "1004": dict(sub="b.y", ct="miscellaneous", df="expert", conf=0.8, att=(1, 1), basis="post_only", sec=40.0, nr=False),
}


def rec(aid, **over):
    r = dict(RUN[aid], **over)
    return {"activity_id": int(aid), "subtopic_id": r["sub"], "content_type": r["ct"], "difficulty": r["df"],
            "confidence": r["conf"], "attempts": {"topic": r["att"][0], "rest": r["att"][1]},
            "needs_review": r["nr"], "time": {"basis": r["basis"]}, "seconds": r["sec"], "model": "m"}


def write_csv(path, header, rows, encoding="utf-8-sig"):
    with open(path, "w", newline="", encoding=encoding) as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


PLAIN_HEADER = ["gold_id", "activity_id", "subtopic_id", "content_type", "difficulty"]
REVIEWED_HEADER = X.REVIEWED_COLUMNS
KEY_HEADER = ["gold_id", "activity_id", "stratum", "cluster", "clustering_subtopic"]


@pytest.fixture
def plain(tmp_path):
    return write_csv(tmp_path / "gold_labels.csv", PLAIN_HEADER, PLAIN_ROWS)


@pytest.fixture
def reviewed(tmp_path):
    return write_csv(tmp_path / "gold_labels_reviewed.csv", REVIEWED_HEADER, REVIEWED_ROWS)


@pytest.fixture
def key_file(tmp_path):
    return write_csv(tmp_path / "gold_key.csv", KEY_HEADER, KEY_ROWS)


def write_run(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def run_file(tmp_path):
    return write_run(tmp_path / "classifications_demo.jsonl", [rec(a) for a in RUN])


@pytest.fixture
def world(plain, key_file, run_file):
    return X.read_label_sets(plain)["labels"], X.read_run(run_file), X.read_key(key_file)


# ============================================================ small helpers

@pytest.mark.parametrize("sub,topic", [("ai-agent-development.ai-agent-development", "ai-agent-development"),
                                       ("other.general", "other"), ("nodot", "nodot"), ("a.b.c", "a")])
def test_topic_of(sub, topic):
    assert X.topic_of(sub) == topic


def test_same_is_none_when_the_gold_label_is_blank():
    assert X._same(None, "x") is None
    assert X._same("x", "x") is True
    assert X._same("x", "y") is False
    assert X._same("x", None) is False


@pytest.mark.parametrize("raw,expected", [("  a ", "a"), ("", None), ("   ", None), (None, None)])
def test_blank_to_none(raw, expected):
    assert X._blank_to_none(raw) == expected


def test_run_name_strips_the_prefix_and_extension():
    assert X.run_name(Path("out/classifications_qwen3-latest.jsonl")) == "qwen3-latest"
    assert X.run_name(Path("out/my_run.jsonl")) == "my_run"
    assert X.run_name(Path("classifications_.jsonl")) == ""


# ============================================================ reading labels

def test_plain_labels_give_one_set(plain):
    sets = X.read_label_sets(plain)
    assert list(sets) == ["labels"]
    assert sets["labels"]["1001"] == {"gold_id": "g01", "subtopic_id": "a.x", "content_type": "tutorial",
                                      "difficulty": "beginner"}
    assert len(sets["labels"]) == 4


def test_blank_cells_become_none(plain):
    lab = X.read_label_sets(plain)["labels"]
    assert lab["1003"]["content_type"] is None and lab["1004"]["difficulty"] is None


def test_reviewed_labels_give_original_and_reviewed(reviewed):
    sets = X.read_label_sets(reviewed)
    assert list(sets) == ["original", "reviewed"]
    assert sets["original"]["1002"]["content_type"] == "tutorial"
    assert sets["reviewed"]["1002"]["content_type"] == "news"
    assert sets["original"]["1003"]["difficulty"] == "intermediate"
    assert sets["reviewed"]["1003"]["difficulty"] == "advanced"
    assert sets["original"]["1001"]["gold_id"] == sets["reviewed"]["1001"]["gold_id"] == "g01"


def test_labels_accept_a_file_without_bom_and_with_spaces(tmp_path):
    p = write_csv(tmp_path / "l.csv", PLAIN_HEADER, [(" g01 ", " 1001 ", " a.x ", "tutorial", "beginner")], encoding="utf-8")
    assert X.read_label_sets(p)["labels"]["1001"]["subtopic_id"] == "a.x"
    assert X.read_label_sets(p)["labels"]["1001"]["gold_id"] == "g01"


def test_labels_missing_column(tmp_path):
    p = write_csv(tmp_path / "l.csv", ["gold_id", "activity_id", "subtopic_id"], [("g01", "1", "a.x")])
    with pytest.raises(CP, match="missing columns .*content_type.*difficulty"):
        X.read_label_sets(p)


def test_reviewed_file_missing_column(tmp_path):
    p = write_csv(tmp_path / "l.csv", [c for c in REVIEWED_HEADER if c != "original_difficulty"], [])
    with pytest.raises(CP, match="missing columns .*original_difficulty"):
        X.read_label_sets(p)


def test_labels_duplicate_activity_id(tmp_path):
    p = write_csv(tmp_path / "l.csv", PLAIN_HEADER, [PLAIN_ROWS[0], PLAIN_ROWS[0]])
    with pytest.raises(CP, match="1001 appears twice"):
        X.read_label_sets(p)


def test_labels_without_rows(tmp_path):
    with pytest.raises(CP, match="no rows"):
        X.read_label_sets(write_csv(tmp_path / "l.csv", PLAIN_HEADER, []))


def test_labels_file_missing(tmp_path):
    with pytest.raises(CP, match="cannot read labels"):
        X.read_label_sets(tmp_path / "nope.csv")


def test_labels_not_utf8(tmp_path):
    p = tmp_path / "l.csv"
    p.write_bytes(b"gold_id,activity_id\n\xff\xfe\n")
    with pytest.raises(CP, match="not valid UTF-8"):
        X.read_label_sets(p)


def test_empty_labels_file(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(CP, match="missing columns"):
        X.read_label_sets(p)


# ============================================================ reading the key and the run

def test_key_none_is_empty():
    assert X.read_key(None) == {}


def test_key_values(key_file):
    k = X.read_key(key_file)
    assert k["1003"] == {"stratum": "stated", "clustering_subtopic": "c.x"}
    assert len(k) == 4


def test_key_missing_column(tmp_path):
    p = write_csv(tmp_path / "k.csv", ["gold_id", "activity_id", "stratum"], [("g01", "1", "random")])
    with pytest.raises(CP, match="key k.csv: missing columns .*clustering_subtopic"):
        X.read_key(p)


def test_key_file_missing(tmp_path):
    with pytest.raises(CP, match="cannot read key"):
        X.read_key(tmp_path / "nope.csv")


def test_run_is_keyed_by_text_activity_id(run_file):
    run = X.read_run(run_file)
    assert sorted(run) == ["1001", "1002", "1003", "1004"]
    assert run["1002"]["subtopic_id"] == "a.z"


def test_run_skips_blank_lines_and_the_last_record_wins(tmp_path):
    p = tmp_path / "r.jsonl"
    first, second = rec("1001"), rec("1001", sub="a.q")
    p.write_text(json.dumps(first) + "\n\n   \n" + json.dumps(second) + "\n", encoding="utf-8")
    run = X.read_run(p)
    assert list(run) == ["1001"] and run["1001"]["subtopic_id"] == "a.q"


def test_run_with_bom(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec("1001")) + "\n", encoding="utf-8-sig")
    assert "1001" in X.read_run(p)


def test_run_invalid_json_names_the_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec("1001")) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(CP, match="line 2: not valid JSON"):
        X.read_run(p)


@pytest.mark.parametrize("line", ['[1, 2]', '{"activity_id": 1}', '{"subtopic_id": "a.x"}', '"text"',
                                  '["activity_id", "subtopic_id"]'])      # a list that merely contains the two names
def test_run_line_that_is_not_a_record(tmp_path, line):
    p = tmp_path / "r.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(CP, match="line 1: not a classification record"):
        X.read_run(p)


def test_run_without_records(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(CP, match="no records"):
        X.read_run(p)


def test_run_file_missing(tmp_path):
    with pytest.raises(CP, match="cannot read run"):
        X.read_run(tmp_path / "nope.jsonl")


def test_run_not_utf8(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(CP, match="not valid UTF-8"):
        X.read_run(p)


# ============================================================ auc

def test_auc_perfect_reversed_and_ties():
    assert X.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert X.auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert X.auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_auc_mixed_by_hand():
    # right: 0.9, 0.7   wrong: 0.8, 0.6   ->  wins 0.9>0.8, 0.9>0.6, 0.7>0.6 = 3 of 4
    assert X.auc([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == 0.75


def test_auc_needs_both_groups():
    assert X.auc([0.9, 0.8], [True, True]) is None
    assert X.auc([0.9, 0.8], [False, False]) is None
    assert X.auc([], []) is None


# ============================================================ one row per post

def rows_by_gold(world):
    labels, run, key = world
    rows, missing = X.post_rows(labels, run, key)
    return {r["gold_id"]: r for r in rows}, missing


def test_post_rows_flags(world):
    rows, missing = rows_by_gold(world)
    assert missing == []
    g1, g2, g3, g4 = (rows[g] for g in ("g01", "g02", "g03", "g04"))
    assert (g1["subtopic_ok"], g1["topic_ok"], g1["content_type_ok"], g1["difficulty_ok"]) == (True, True, True, True)
    assert (g2["subtopic_ok"], g2["topic_ok"], g2["content_type_ok"], g2["difficulty_ok"]) == (False, True, False, False)
    assert (g3["subtopic_ok"], g3["topic_ok"], g3["content_type_ok"], g3["difficulty_ok"]) == (False, False, None, False)
    assert (g4["subtopic_ok"], g4["topic_ok"], g4["content_type_ok"], g4["difficulty_ok"]) == (True, True, True, None)


def test_post_rows_clustering_flags(world):
    rows, _ = rows_by_gold(world)
    assert [rows[g]["clustering_subtopic_ok"] for g in ("g01", "g02", "g03", "g04")] == [True, True, False, False]
    assert [rows[g]["clustering_topic_ok"] for g in ("g01", "g02", "g03", "g04")] == [True, True, False, True]
    assert rows["g04"]["clustering_subtopic"] == "b.z"


def test_post_rows_difficulty_delta(world):
    rows, _ = rows_by_gold(world)
    assert rows["g01"]["difficulty_delta"] == 0
    assert rows["g02"]["difficulty_delta"] == 1          # advanced vs intermediate
    assert rows["g03"]["difficulty_delta"] == -2         # beginner vs advanced
    assert rows["g04"]["difficulty_delta"] is None       # blank gold difficulty


def test_post_rows_delta_is_none_for_an_unknown_predicted_level(world):
    labels, run, key = world
    run = {**run, "1001": {**run["1001"], "difficulty": "easy"}}
    rows, _ = X.post_rows(labels, run, key)
    assert next(r for r in rows if r["gold_id"] == "g01")["difficulty_delta"] is None


def test_post_rows_copy_the_run_details(world):
    rows, _ = rows_by_gold(world)
    g2 = rows["g02"]
    assert (g2["confidence"], g2["needs_review"], g2["attempts_topic"], g2["attempts_rest"]) == (0.6, True, 1, 2)
    assert (g2["basis"], g2["seconds"], g2["stratum"]) == ("stated_in_post", 20.0, "random")
    assert (g2["label_subtopic"], g2["pred_subtopic"]) == ("a.y", "a.z")
    assert (g2["label_content_type"], g2["pred_content_type"]) == ("news", "miscellaneous")
    assert (g2["label_difficulty"], g2["pred_difficulty"]) == ("intermediate", "advanced")


def test_post_rows_without_a_key(world):
    labels, run, _ = world
    rows, _ = X.post_rows(labels, run, {})
    assert all(r["clustering_subtopic"] is None and r["clustering_subtopic_ok"] is None
               and r["clustering_topic_ok"] is None and r["stratum"] == "" for r in rows)


def test_post_rows_report_posts_missing_from_the_run(world):
    labels, run, key = world
    run = {k: v for k, v in run.items() if k not in ("1002", "1004")}
    rows, missing = X.post_rows(labels, run, key)
    assert missing == ["g02", "g04"] and [r["gold_id"] for r in rows] == ["g01", "g03"]


def test_post_rows_with_a_blank_gold_subtopic(world):
    labels, run, key = world
    labels = {**labels, "1001": {**labels["1001"], "subtopic_id": None}}
    rows, _ = X.post_rows(labels, run, key)
    r = next(r for r in rows if r["gold_id"] == "g01")
    assert r["subtopic_ok"] is None and r["topic_ok"] is None
    assert r["clustering_subtopic_ok"] is None and r["clustering_topic_ok"] is None   # nothing to compare with


def test_post_rows_record_without_optional_fields(world):
    labels, _, key = world
    bare = {"1001": {"activity_id": 1001, "subtopic_id": "a.x"}}
    rows, _ = X.post_rows({"1001": labels["1001"]}, bare, key)
    r = rows[0]
    assert r["pred_content_type"] is None and r["content_type_ok"] is False and r["pred_difficulty"] is None
    assert r["difficulty_delta"] is None and r["confidence"] is None and r["needs_review"] is False
    assert r["attempts_topic"] is None and r["basis"] is None and r["seconds"] is None


# ============================================================ tally and summarise

def test_tally_counts_true_and_skips_none():
    rows = [{"f": True}, {"f": False}, {"f": None}, {"f": True}]
    assert X.tally(rows, "f") == {"right": 2, "total": 3}
    assert X.tally([], "f") == {"right": 0, "total": 0}


@pytest.fixture
def summary(world):
    labels, run, key = world
    rows, missing = X.post_rows(labels, run, key)
    return X.summarise(rows, missing)


def test_summary_lens_counts(summary):
    assert summary["posts"] == 4 and summary["missing_from_run"] == []
    assert summary["subtopic"] == {"right": 2, "total": 4}
    assert summary["topic"] == {"right": 3, "total": 4}
    assert summary["content_type"] == {"right": 2, "total": 3}      # g03's type is blank
    assert summary["difficulty"] == {"right": 1, "total": 3}        # g04's difficulty is blank


def test_summary_clustering_counts(summary):
    assert summary["clustering"]["subtopic"] == {"right": 2, "total": 4}
    assert summary["clustering"]["topic"] == {"right": 3, "total": 4}


def test_summary_by_stratum(summary):
    s = summary["by_stratum"]
    assert sorted(s) == ["generic", "random", "stated"]
    assert s["random"] == {"posts": 2, "subtopic": {"right": 1, "total": 2}, "topic": {"right": 2, "total": 2},
                           "content_type": {"right": 1, "total": 2}, "difficulty": {"right": 1, "total": 2}}
    assert s["stated"]["posts"] == 1 and s["stated"]["content_type"] == {"right": 0, "total": 0}
    assert s["generic"]["difficulty"] == {"right": 0, "total": 0}


def test_summary_stratum_without_a_key_is_named_none(world):
    labels, run, _ = world
    rows, missing = X.post_rows(labels, run, {})
    assert list(X.summarise(rows, missing)["by_stratum"]) == ["(none)"]


def test_summary_confidence(summary):
    c = summary["confidence"]
    # right: g01 0.9, g04 0.8 -> 0.85;   wrong: g02 0.6, g03 0.5 -> 0.55;   every right beats every wrong
    assert c["mean_when_right"] == pytest.approx(0.85) and c["mean_when_wrong"] == pytest.approx(0.55)
    assert c["auc_subtopic"] == 1.0


def test_summary_confidence_with_nothing_wrong(world):
    labels, run, key = world
    rows, missing = X.post_rows({k: v for k, v in labels.items() if k in ("1001", "1004")}, run, key)
    c = X.summarise(rows, missing)["confidence"]
    assert c["mean_when_wrong"] is None and c["auc_subtopic"] is None and c["mean_when_right"] == pytest.approx(0.85)


def test_summary_confidence_ignores_non_numeric_values(world):
    labels, run, key = world
    run = {**run, "1002": {**run["1002"], "confidence": None}}
    rows, missing = X.post_rows(labels, run, key)
    assert X.summarise(rows, missing)["confidence"]["mean_when_wrong"] == pytest.approx(0.5)


def test_summary_content_type_confusion(summary):
    assert summary["content_type_confusion"] == {"miscellaneous": {"miscellaneous": 1},
                                                 "news": {"miscellaneous": 1},
                                                 "tutorial": {"tutorial": 1}}


def test_summary_difficulty_delta(summary):
    assert summary["difficulty_delta"] == {"-2": 1, "0": 1, "1": 1}


def test_summary_run_statistics(summary):
    assert summary["retried_posts"] == 2        # g02 (rest asked twice) and g03 (topic asked twice)
    assert summary["needs_review"] == 1
    assert summary["basis"] == {"post_only": 3, "stated_in_post": 1}
    assert summary["seconds"] == {"total": 100.0, "mean": 25.0}


def test_summary_seconds_missing(world):
    labels, run, key = world
    run = {k: {kk: vv for kk, vv in v.items() if kk != "seconds"} for k, v in run.items()}
    rows, missing = X.post_rows(labels, run, key)
    assert X.summarise(rows, missing)["seconds"] == {"total": 0, "mean": None}


def test_summary_of_nothing():
    s = X.summarise([], ["g01"])
    assert s["posts"] == 0 and s["missing_from_run"] == ["g01"] and s["subtopic"] == {"right": 0, "total": 0}
    assert s["by_stratum"] == {} and s["confidence"]["auc_subtopic"] is None


# ============================================================ compare, format, write

def test_compare_scores_every_run_against_every_label_set(reviewed, run_file, key_file):
    sets = X.read_label_sets(reviewed)
    other = {k: {**v, "subtopic_id": "zzz.zzz"} for k, v in X.read_run(run_file).items()}
    report, rows = X.compare(sets, {"demo": X.read_run(run_file), "bad": other}, X.read_key(key_file))
    assert list(report) == ["original", "reviewed"] and list(report["reviewed"]) == ["demo", "bad"]
    assert len(rows) == 2 * 2 * 4
    assert {(r["label_set"], r["run"]) for r in rows} == {(s, n) for s in ("original", "reviewed") for n in ("demo", "bad")}
    # the label sets differ in g02's type: original "tutorial", reviewed "news"; the classifier says miscellaneous for both
    assert report["original"]["demo"]["content_type"] == {"right": 2, "total": 3}
    assert report["reviewed"]["demo"]["content_type"] == {"right": 2, "total": 3}
    assert report["original"]["demo"]["difficulty"] == {"right": 1, "total": 3}
    assert report["reviewed"]["demo"]["difficulty"] == {"right": 1, "total": 3}
    assert report["original"]["bad"]["subtopic"]["right"] == 0


def test_compare_label_changes_move_the_score(tmp_path, run_file, key_file):
    # original g03 difficulty is "beginner" here, so the classifier (beginner) is right on the original but not the reviewed
    rows = [r[:6] + ("beginner", "advanced") if r[0] == "g03" else r for r in REVIEWED_ROWS]
    p = write_csv(tmp_path / "rev.csv", REVIEWED_HEADER, rows)
    report, _ = X.compare(X.read_label_sets(p), {"demo": X.read_run(run_file)}, X.read_key(key_file))
    assert report["original"]["demo"]["difficulty"]["right"] == report["reviewed"]["demo"]["difficulty"]["right"] + 1


def test_format_report_shows_fractions_and_baseline(world):
    labels, run, key = world
    report, _ = X.compare({"labels": labels}, {"demo": run}, key)
    text = X.format_report(report)
    assert "label set: labels (4 posts)" in text
    assert re.search(r"clustering \(baseline\)\s+2/4\s+3/4\s+-\s+-\n", text)
    assert re.search(r"\ndemo\s+2/4\s+3/4\s+2/3\s+1/3\n", text)
    assert "AUC 1.00" in text and "0.85 when the subtopic is right, 0.55 when wrong" in text
    assert "re-asked posts 2/4, needs_review 1" in text
    assert "type you chose news: classifier said {'miscellaneous': 1}" in text
    assert "NOT IN THIS RUN" not in text


def test_format_report_without_a_key_has_no_baseline_row(world):
    labels, run, _ = world
    report, _ = X.compare({"labels": labels}, {"demo": run}, {})
    assert "clustering (baseline)" not in X.format_report(report)


def test_format_report_lists_missing_posts_and_dashes(world):
    labels, run, key = world
    report, _ = X.compare({"labels": labels}, {"demo": {"1001": run["1001"]}}, key)
    text = X.format_report(report)
    assert "NOT IN THIS RUN: g02, g03, g04" in text


def test_format_report_dash_for_empty_numbers():
    report = {"s": {"r": X.summarise([], [])}}
    text = X.format_report(report)
    assert "AUC -" in text and "predicted level minus yours: -" in text


def test_format_report_with_two_label_sets(reviewed, run_file, key_file):
    report, _ = X.compare(X.read_label_sets(reviewed), {"demo": X.read_run(run_file)}, X.read_key(key_file))
    text = X.format_report(report)
    assert text.index("label set: original") < text.index("label set: reviewed")


def test_write_outputs(tmp_path, world):
    labels, run, key = world
    report, rows = X.compare({"labels": labels}, {"demo": run}, key)
    out = tmp_path / "new" / "dir"
    paths = X.write_outputs(out, report, rows)
    assert [p.name for p in paths] == [X.REPORT_FILE, X.POSTS_FILE] and all(p.exists() for p in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == report
    with open(paths[1], newline="", encoding="utf-8-sig") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == X.POST_COLUMNS and len(got) == 4
    assert got[1]["gold_id"] == "g02" and got[1]["pred_subtopic"] == "a.z" and got[1]["label_set"] == "labels"
    assert got[1]["difficulty_delta"] == "1" and got[1]["run"] == "demo"
    assert got[2]["content_type_ok"] == ""            # blank gold type is written empty, not "None"


def test_write_outputs_overwrites_derived_files(tmp_path, world):
    labels, run, key = world
    report, rows = X.compare({"labels": labels}, {"demo": run}, key)
    X.write_outputs(tmp_path, {"old": 1}, [])
    X.write_outputs(tmp_path, report, rows)
    assert json.loads((tmp_path / X.REPORT_FILE).read_text(encoding="utf-8")) == report


# ============================================================ command line

def cli(tmp_path, *args):
    return X.main([*map(str, args), "--out-dir", str(tmp_path / "out")])


def test_main_runs_end_to_end(tmp_path, reviewed, key_file, run_file, capsys):
    assert cli(tmp_path, "--labels", reviewed, "--key", key_file, "--runs", run_file) == 0
    out = capsys.readouterr().out
    assert "label set: original" in out and "label set: reviewed" in out and "demo" in out
    assert (tmp_path / "out" / X.REPORT_FILE).exists() and (tmp_path / "out" / X.POSTS_FILE).exists()


def test_main_works_without_a_key(tmp_path, plain, run_file, capsys):
    assert cli(tmp_path, "--labels", plain, "--runs", run_file) == 0
    assert "clustering (baseline)" not in capsys.readouterr().out


def test_main_compares_several_runs(tmp_path, plain, run_file, capsys):
    second = write_run(tmp_path / "classifications_other.jsonl", [rec(a, sub="b.q") for a in RUN])
    assert cli(tmp_path, "--labels", plain, "--runs", run_file, second) == 0
    report = json.loads((tmp_path / "out" / X.REPORT_FILE).read_text(encoding="utf-8"))
    assert list(report["labels"]) == ["demo", "other"]
    assert report["labels"]["other"]["subtopic"]["right"] == 0


def test_main_problem_with_labels(tmp_path, run_file, capsys):
    assert cli(tmp_path, "--labels", tmp_path / "nope.csv", "--runs", run_file) == 1
    assert "PROBLEMS" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_main_run_with_none_of_the_gold_posts(tmp_path, plain, capsys):
    other = write_run(tmp_path / "classifications_x.jsonl", [dict(rec("1001"), activity_id=999)])
    assert cli(tmp_path, "--labels", plain, "--runs", other) == 1
    assert "has none of the gold posts" in capsys.readouterr().out


def test_main_run_with_some_gold_posts_is_fine(tmp_path, plain, capsys):
    part = write_run(tmp_path / "classifications_part.jsonl", [rec("1001")])
    assert cli(tmp_path, "--labels", plain, "--runs", part) == 0
    assert "NOT IN THIS RUN: g02, g03, g04" in capsys.readouterr().out


def test_main_duplicate_run_names(tmp_path, plain, run_file, capsys):
    sub = tmp_path / "sub"
    sub.mkdir()
    twin = write_run(sub / "classifications_demo.jsonl", [rec(a) for a in RUN])
    assert cli(tmp_path, "--labels", plain, "--runs", run_file, twin) == 1
    assert "two runs are called 'demo'" in capsys.readouterr().out


def test_main_key_that_misses_gold_posts(tmp_path, plain, run_file, capsys):
    short = write_csv(tmp_path / "short_key.csv", KEY_HEADER, KEY_ROWS[:1])
    assert cli(tmp_path, "--labels", plain, "--key", short, "--runs", run_file) == 1
    assert "does not list gold posts" in capsys.readouterr().out


def test_main_bad_run_file(tmp_path, plain, capsys):
    bad = tmp_path / "classifications_bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    assert cli(tmp_path, "--labels", plain, "--runs", bad) == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_main_requires_its_options(tmp_path, plain):
    with pytest.raises(SystemExit):
        X.main(["--labels", str(plain)])


# ============================================================ extra edge cases found while planning the mutation check

def test_blank_clustering_cell_in_the_key_means_no_opinion(world):
    labels, run, key = world
    key = {**key, "1001": {"stratum": "random", "clustering_subtopic": ""}}
    rows, _ = X.post_rows(labels, run, key)
    r = next(r for r in rows if r["gold_id"] == "g01")
    assert r["clustering_subtopic"] is None and r["clustering_subtopic_ok"] is None and r["clustering_topic_ok"] is None


def test_format_report_shows_a_dash_for_lenses_with_no_labels():
    text = X.format_report({"s": {"r": X.summarise([], [])}})
    assert re.search(r"\nr\s+-\s+-\s+-\s+-\n", text)       # the table row: all four lenses are dashes, not 0/0


def test_posts_csv_columns_match_the_row_fields(world):
    labels, run, key = world
    _, rows = X.compare({"labels": labels}, {"demo": run}, key)
    assert set(rows[0]) == set(X.POST_COLUMNS) and len(X.POST_COLUMNS) == len(set(X.POST_COLUMNS))


def test_posts_csv_starts_with_a_bom_so_excel_reads_utf8(tmp_path, world):
    labels, run, key = world
    report, rows = X.compare({"labels": labels}, {"demo": run}, key)
    X.write_outputs(tmp_path, report, rows)
    assert (tmp_path / X.POSTS_FILE).read_bytes().startswith(b"\xef\xbb\xbf")
