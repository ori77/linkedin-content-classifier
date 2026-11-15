"""Tests for pipeline/step17_export.py (Step 17): the CSV and JSON exports.

Uses the made-up database of tests/dash_fixture.py (built with the real loader). The main promises:
every row of every export carries the post link and the resource links; your corrected labels
show next to the model's; the CSV opens correctly in Excel; a failed write never leaves half a file.
"""
import csv
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))

import dash_fixture as F  # noqa: E402
import dashboard_data as D  # noqa: E402
import step17_export as X  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return F.build(tmp_path)


@pytest.fixture
def con(db):
    c = D.connect(db)
    yield c
    c.close()


@pytest.fixture
def rows(con):
    return D.load_library(con)


def by_aid(records, aid):
    return next(r for r in records if r["activity_id"] == str(aid))


def read_csv(data: bytes):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def set_review_time(db, review_id, stamp):
    c = sqlite3.connect(db)
    with c:
        c.execute("UPDATE reviews SET created_at = ? WHERE id = ?", (stamp, review_id))
    c.close()


# ============================================================ the columns

def test_every_export_has_the_link_columns():
    for columns in (X.LIBRARY_COLUMNS, X.CORRECTION_COLUMNS):
        assert set(X.LINK_COLUMNS) <= set(columns)
    assert X.LINK_COLUMNS == ("post_url", "resource_link", "resource_links", "resource_link_count")


def test_column_lists_have_no_duplicates():
    assert len(set(X.LIBRARY_COLUMNS)) == len(X.LIBRARY_COLUMNS)
    assert len(set(X.CORRECTION_COLUMNS)) == len(X.CORRECTION_COLUMNS)


def test_records_have_exactly_the_listed_columns_in_order(rows):
    lib = X.library_records(rows)
    assert len(lib) == 6 and all(list(r) == list(X.LIBRARY_COLUMNS) for r in lib)


# ============================================================ library records

def test_one_record_per_classified_post_in_the_order_given(rows):
    recs = X.library_records(rows)
    assert [r["activity_id"] for r in recs] == ["1", "2", "3", "4", "5", "6"]       # post 7 has no text: not in the library
    assert [r["activity_id"] for r in X.library_records(list(reversed(rows)))] == ["6", "5", "4", "3", "2", "1"]
    assert X.library_records([]) == []


def test_activity_id_is_text(rows):
    assert all(isinstance(r["activity_id"], str) for r in X.library_records(rows))


def test_every_library_record_has_the_post_link(rows):
    for r in X.library_records(rows):
        assert r["post_url"] == F.url_of(int(r["activity_id"]))


def test_resource_links_are_all_there(rows):
    recs = X.library_records(rows)
    one = by_aid(recs, 1)
    assert one["resource_links"] == ["https://lnkd.in/a1", "https://example.com/two", "https://lnkd.in/a3"]
    assert one["resource_link"] == "https://lnkd.in/a1" and one["resource_link_count"] == 3
    none = by_aid(recs, 2)
    assert none["resource_links"] == [] and none["resource_link"] is None and none["resource_link_count"] == 0
    six = by_aid(recs, 6)
    assert six["resource_links"] == ["https://lnkd.in/f1", "https://example.com/f(2)"] and six["resource_link_count"] == 2


def test_library_values(rows):
    r = by_aid(X.library_records(rows), 3)
    assert (r["title"], r["author"], r["saved_at"], r["post_date"]) == ("Interview prep", "Cy", "2026-09-01 10:00:00", "2026-10-01")
    assert (r["topic_id"], r["topic_name"], r["subtopic_id"], r["subtopic_name"]) == (
        "careers", "Career Development", "careers.interviews", "Interviews")
    assert (r["difficulty"], r["content_type"]) == ("intermediate", "interview_question_set")
    assert (r["cover_minutes"], r["cover_bucket"], r["understand_minutes"], r["understand_bucket"], r["time_basis"]) == (
        40, "15-60m", 40, "15-60m", "post_only")
    assert (r["alignment_score"], r["alignment_band"], r["summary"]) == (3.0, "Low relevance", "Summary of post 3.")
    assert r["confidence"] == 0.61 and r["model"] == "gemma4:e4b" and r["taxonomy_version"] == "v1"
    assert r["status"] == "needs_review" and r["model_flagged"] is True and r["reviewed"] is False
    assert r["review_reasons"] == ["low_confidence", "topic_disagrees_with_clustering"]
    assert r["changed_fields"] == [] and r["review_note"] is None
    assert (r["model_subtopic_id"], r["model_difficulty"], r["model_content_type"]) == (
        "careers.interviews", "intermediate", "interview_question_set")


def test_an_unflagged_post_is_not_marked_flagged(rows):
    r = by_aid(X.library_records(rows), 1)
    assert r["model_flagged"] is False and r["status"] == "auto_accepted" and r["review_reasons"] == []


def test_a_missing_save_date_stays_empty(rows):
    assert by_aid(X.library_records(rows), 5)["saved_at"] is None


def test_the_library_shows_your_fix_next_to_the_models_label(con):
    D.save_fix(con, "3", {"difficulty": "advanced", "subtopic_id": "rag.chunking"}, note="too easy")
    r = by_aid(X.library_records(D.load_library(con)), 3)
    assert r["difficulty"] == "advanced" and r["model_difficulty"] == "intermediate"
    assert r["subtopic_id"] == "rag.chunking" and r["model_subtopic_id"] == "careers.interviews"
    assert (r["topic_id"], r["topic_name"], r["subtopic_name"]) == ("rag", "Retrieval-Augmented Generation", "Chunking")
    assert r["content_type"] == r["model_content_type"] == "interview_question_set"
    assert r["reviewed"] is True and r["status"] == "reviewed" and r["review_note"] == "too easy"
    assert sorted(r["changed_fields"]) == ["difficulty", "subtopic_id"]
    assert r["model_flagged"] is True                       # the model's own flag is kept


def test_an_approval_marks_the_post_reviewed_without_changed_fields(con):
    D.approve_review(con, "3")
    r = by_aid(X.library_records(D.load_library(con)), 3)
    assert r["reviewed"] is True and r["changed_fields"] == [] and r["difficulty"] == r["model_difficulty"]


def test_a_post_with_a_special_title_is_kept_as_it_is(rows):
    assert by_aid(X.library_records(rows), 6)["title"] == "Tips & $tricks* 100%"


# ============================================================ corrections records

def test_no_reviews_no_correction_records(rows):
    assert X.correction_records(rows) == []


def test_correction_records_have_exactly_the_listed_columns(con):
    D.approve_review(con, "3")
    recs = X.correction_records(D.load_library(con))
    assert all(list(r) == list(X.CORRECTION_COLUMNS) for r in recs)


def test_an_approval_record(con):
    rid = D.approve_review(con, "3", note="  fine  ", use_as_example=False)
    (r,) = X.correction_records(D.load_library(con))
    assert r["activity_id"] == "3" and r["title"] == "Interview prep" and r["author"] == "Cy"
    assert r["review_id"] == rid and r["decision"] == "approved" and r["changed_fields"] == []
    assert r["note"] == "fine" and r["use_as_example"] is False
    assert (r["final_subtopic_id"], r["final_difficulty"], r["final_content_type"]) == (
        r["model_subtopic_id"], r["model_difficulty"], r["model_content_type"]) == (
        "careers.interviews", "intermediate", "interview_question_set")
    assert r["reviewed_at"].endswith("+00:00")


def test_a_fix_record(con):
    D.save_fix(con, "3", {"difficulty": "advanced", "content_type": "news"}, note="n", use_as_example=True)
    (r,) = X.correction_records(D.load_library(con))
    assert r["decision"] == "corrected" and sorted(r["changed_fields"]) == ["content_type", "difficulty"]
    assert (r["model_difficulty"], r["final_difficulty"]) == ("intermediate", "advanced")
    assert (r["model_content_type"], r["final_content_type"]) == ("interview_question_set", "news")
    assert (r["model_subtopic_id"], r["final_subtopic_id"]) == ("careers.interviews", "careers.interviews")
    assert r["use_as_example"] is True and r["note"] == "n"
    assert r["summary"] == "Summary of post 3." and r["confidence"] == 0.61 and r["model_flagged"] is True
    assert r["model"] == "gemma4:e4b" and r["taxonomy_version"] == "v1"
    assert r["review_reasons"] == ["low_confidence", "topic_disagrees_with_clustering"]


def test_a_subtopic_fix_shows_the_final_subtopic(con):
    D.save_fix(con, "6", {"subtopic_id": "rag.chunking"})
    (r,) = X.correction_records(D.load_library(con))
    assert (r["model_subtopic_id"], r["final_subtopic_id"]) == ("rag.evaluation", "rag.chunking")


def test_every_correction_record_has_its_links(con):
    for aid in ("1", "3", "6", "2"):
        D.approve_review(con, aid)
    recs = X.correction_records(D.load_library(con))
    assert len(recs) == 4
    for r in recs:
        assert r["post_url"] == F.url_of(int(r["activity_id"]))
    assert by_aid(recs, 1)["resource_links"] == ["https://lnkd.in/a1", "https://example.com/two", "https://lnkd.in/a3"]
    assert by_aid(recs, 1)["resource_link"] == "https://lnkd.in/a1" and by_aid(recs, 1)["resource_link_count"] == 3
    assert by_aid(recs, 2)["resource_link"] is None and by_aid(recs, 2)["resource_links"] == []


def test_only_the_newest_review_of_a_post_is_exported(con, db):
    first = D.save_fix(con, "3", {"difficulty": "advanced"})[0]
    second = D.approve_review(con, "3")
    set_review_time(db, first, "2026-11-14T10:00:00+00:00")
    set_review_time(db, second, "2026-11-14T11:00:00+00:00")
    (r,) = X.correction_records(D.load_library(con))
    assert r["review_id"] == second and r["decision"] == "approved"


def test_only_reviewed_posts_are_exported(con):
    D.approve_review(con, "4")
    recs = X.correction_records(D.load_library(con))
    assert [r["activity_id"] for r in recs] == ["4"]


def test_corrections_are_oldest_review_first_and_ties_go_by_post_id(con, db):
    # (the rows are handed over in reverse, so a tie is only broken by the post id, not by the input order)
    ids = {aid: D.approve_review(con, aid) for aid in ("6", "1", "4", "3")}
    set_review_time(db, ids["6"], "2026-11-14T10:00:00+00:00")
    set_review_time(db, ids["1"], "2026-11-14T11:00:00+00:00")
    set_review_time(db, ids["4"], "2026-11-14T09:00:00+00:00")
    set_review_time(db, ids["3"], "2026-11-14T09:00:00+00:00")
    recs = X.correction_records(list(reversed(D.load_library(con))))
    assert [r["activity_id"] for r in recs] == ["3", "4", "6", "1"]


def test_load_library_gives_the_review_details(con, db):
    rid = D.approve_review(con, "3", use_as_example=False)
    r = next(x for x in D.load_library(con) if x["activity_id"] == 3)
    assert r["review_id"] == rid and r["use_as_example"] is False and r["reviewed_at"].endswith("+00:00")
    other = next(x for x in D.load_library(con) if x["activity_id"] == 4)
    assert other["review_id"] is None and other["reviewed_at"] is None and other["use_as_example"] is None
    rid2 = D.approve_review(con, "4", use_as_example=True)
    assert next(x for x in D.load_library(con) if x["activity_id"] == 4)["use_as_example"] is True
    assert next(x for x in D.load_library(con) if x["activity_id"] == 4)["review_id"] == rid2


# ============================================================ csv_cell

def test_csv_cell_basic_values():
    assert X.csv_cell(None) == "" and X.csv_cell("") == ""
    assert X.csv_cell(True) == "true" and X.csv_cell(False) == "false"
    assert X.csv_cell(7) == "7" and X.csv_cell(0) == "0" and X.csv_cell(0.5) == "0.5" and X.csv_cell(10.0) == "10.0"
    assert X.csv_cell("plain") == "plain" and X.csv_cell("a, b \"c\"") == "a, b \"c\""


def test_csv_cell_lists():
    assert X.csv_cell(["a", "b", "c"]) == "a | b | c"
    assert X.csv_cell(["only"]) == "only" and X.csv_cell([]) == ""


@pytest.mark.parametrize("first", list("=+-@") + ["\t", "\r"])
def test_csv_cell_defuses_formulas(first):
    assert X.csv_cell(f"{first}SUM(A1)") == f"'{first}SUM(A1)"


def test_csv_cell_leaves_other_starts_alone():
    for text in ("'quoted", " =spaced", "x=1", "http://a.b/?q=1", "7 ways", "at-end@"):
        assert X.csv_cell(text) == text


def test_csv_cell_defuses_a_list_that_starts_with_a_formula():
    assert X.csv_cell(["=1", "b"]) == "'=1 | b"


def test_csv_cell_does_not_touch_numbers():
    assert X.csv_cell(-5) == "-5" and X.csv_cell(-0.25) == "-0.25"


# ============================================================ csv bytes

def test_csv_starts_with_a_byte_order_mark_and_the_header(rows):
    data = X.to_csv_bytes(X.library_records(rows), X.LIBRARY_COLUMNS)
    assert data.startswith(b"\xef\xbb\xbf")
    assert data[3:].decode("utf-8").splitlines()[0] == ",".join(X.LIBRARY_COLUMNS)


def test_csv_round_trip_keeps_every_row_and_every_link(rows):
    recs = X.library_records(rows)
    back = read_csv(X.to_csv_bytes(recs, X.LIBRARY_COLUMNS))
    assert len(back) == 6 and [b["activity_id"] for b in back] == ["1", "2", "3", "4", "5", "6"]
    for rec, b in zip(recs, back):
        assert b["post_url"] == rec["post_url"] and b["post_url"].startswith("https://")
        assert b["resource_link"] == (rec["resource_link"] or "")
        assert b["resource_link_count"] == str(rec["resource_link_count"])
    assert back[0]["resource_links"] == "https://lnkd.in/a1 | https://example.com/two | https://lnkd.in/a3"
    assert back[1]["resource_links"] == "" and back[1]["resource_link"] == ""
    assert back[5]["resource_links"] == "https://lnkd.in/f1 | https://example.com/f(2)"


def test_csv_values_are_written_as_documented(rows):
    back = read_csv(X.to_csv_bytes(X.library_records(rows), X.LIBRARY_COLUMNS))
    r = back[2]
    assert r["model_flagged"] == "true" and r["reviewed"] == "false" and r["alignment_score"] == "3.0"
    assert r["confidence"] == "0.61" and r["cover_minutes"] == "40" and r["changed_fields"] == "" and r["review_note"] == ""
    assert back[4]["saved_at"] == ""
    assert r["review_reasons"] == "low_confidence | topic_disagrees_with_clustering"


def test_csv_survives_commas_quotes_and_line_breaks_in_text(rows):
    recs = X.library_records(rows)
    recs[0] = dict(recs[0], summary='He said "hi", then\nleft.\r\nBye', title="café 日本語 🙂")
    back = read_csv(X.to_csv_bytes(recs, X.LIBRARY_COLUMNS))
    assert back[0]["summary"] == 'He said "hi", then\nleft.\r\nBye' and back[0]["title"] == "café 日本語 🙂"
    assert len(back) == 6


def test_csv_defuses_a_formula_in_a_title(rows):
    recs = X.library_records(rows)
    recs[0] = dict(recs[0], title="=HYPERLINK(\"http://x\")", author="@someone", summary="-1+1")
    back = read_csv(X.to_csv_bytes(recs, X.LIBRARY_COLUMNS))
    assert back[0]["title"].startswith("'=") and back[0]["author"] == "'@someone" and back[0]["summary"] == "'-1+1"


def test_csv_for_no_records_is_just_the_header():
    data = X.to_csv_bytes([], X.CORRECTION_COLUMNS)
    assert data.decode("utf-8-sig").strip() == ",".join(X.CORRECTION_COLUMNS) and read_csv(data) == []


def test_correction_csv_round_trip(con):
    D.save_fix(con, "6", {"difficulty": "advanced"}, note="a, \"b\"")
    D.approve_review(con, "1")
    back = read_csv(X.to_csv_bytes(X.correction_records(D.load_library(con)), X.CORRECTION_COLUMNS))
    assert {b["activity_id"] for b in back} == {"1", "6"}
    six = next(b for b in back if b["activity_id"] == "6")
    assert six["decision"] == "corrected" and six["changed_fields"] == "difficulty" and six["note"] == "a, \"b\""
    assert six["use_as_example"] == "true" and six["resource_links"] == "https://lnkd.in/f1 | https://example.com/f(2)"
    assert six["post_url"] == F.url_of(6)
    one = next(b for b in back if b["activity_id"] == "1")
    assert one["decision"] == "approved" and one["resource_link_count"] == "3"


# ============================================================ json bytes

def test_json_has_the_envelope_and_all_rows(rows):
    data = X.to_json_bytes(X.library_records(rows), "library", "2026-11-15T10:00:00+00:00")
    doc = json.loads(data.decode("utf-8"))
    assert list(doc) == ["kind", "exported_at", "count", "rows"]
    assert doc["kind"] == "library" and doc["exported_at"] == "2026-11-15T10:00:00+00:00" and doc["count"] == 6
    assert len(doc["rows"]) == 6 and data.endswith(b"}\n") and not data.startswith(b"\xef\xbb\xbf")


def test_json_keeps_types_and_every_link(rows):
    doc = json.loads(X.to_json_bytes(X.library_records(rows), "library", "t"))
    r = doc["rows"][0]
    assert r["activity_id"] == "1" and r["resource_links"] == ["https://lnkd.in/a1", "https://example.com/two", "https://lnkd.in/a3"]
    assert r["resource_link_count"] == 3 and r["model_flagged"] is False and r["reviewed"] is False
    assert r["post_url"] == F.url_of(1) and r["cover_minutes"] == 30 and r["alignment_score"] == 10.0
    assert doc["rows"][1]["resource_link"] is None and doc["rows"][1]["resource_links"] == []
    assert doc["rows"][4]["saved_at"] is None
    for row in doc["rows"]:
        assert row["post_url"].startswith("https://")


def test_json_is_not_changed_for_formulas_and_keeps_unicode_readable():
    recs = [{"title": "=cmd", "x": "日本語"}]
    data = X.to_json_bytes(recs, "library", "t")
    assert json.loads(data)["rows"][0] == {"title": "=cmd", "x": "日本語"} and "日本語".encode("utf-8") in data


def test_json_is_indented_for_reading():
    assert b'\n  "kind": "corrections"' in X.to_json_bytes([], "corrections", "t")
    doc = json.loads(X.to_json_bytes([], "corrections", "t"))
    assert doc["count"] == 0 and doc["rows"] == []


# ============================================================ writing files

def test_write_atomic_writes_and_replaces(tmp_path):
    path = tmp_path / "f.bin"
    X.write_atomic(path, b"one")
    assert path.read_bytes() == b"one"
    X.write_atomic(path, b"two")
    assert path.read_bytes() == b"two" and sorted(p.name for p in tmp_path.iterdir()) == ["f.bin"]


def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    X.write_atomic(path, b"old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(X.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        X.write_atomic(path, b"new")
    assert path.read_bytes() == b"old" and sorted(p.name for p in tmp_path.iterdir()) == ["f.bin"]


def test_export_files_writes_all_four_files(rows, tmp_path):
    out = tmp_path / "exports"
    written = X.export_files(rows, out, exported_at="2026-11-15T10:00:00+00:00")
    assert sorted(p.name for p, _ in written) == ["corrections.csv", "corrections.json", "library.csv", "library.json"]
    assert {p.name: n for p, n in written} == {"library.csv": 6, "library.json": 6, "corrections.csv": 0, "corrections.json": 0}
    assert sorted(p.name for p in out.iterdir()) == ["corrections.csv", "corrections.json", "library.csv", "library.json"]
    assert json.loads((out / "library.json").read_text(encoding="utf-8"))["exported_at"] == "2026-11-15T10:00:00+00:00"
    assert json.loads((out / "corrections.json").read_text(encoding="utf-8"))["count"] == 0


def test_the_files_contain_what_the_functions_return(rows, tmp_path):
    X.export_files(rows, tmp_path)
    assert (tmp_path / "library.csv").read_bytes() == X.to_csv_bytes(X.library_records(rows), X.LIBRARY_COLUMNS)
    assert (tmp_path / "corrections.csv").read_bytes() == X.to_csv_bytes([], X.CORRECTION_COLUMNS)
    doc = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    assert doc["kind"] == "library" and doc["rows"] == json.loads(json.dumps(X.library_records(rows)))


def test_corrections_files_hold_the_reviews(con, tmp_path):
    D.approve_review(con, "3")
    X.export_files(D.load_library(con), tmp_path)
    assert json.loads((tmp_path / "corrections.json").read_text(encoding="utf-8"))["count"] == 1
    assert len(read_csv((tmp_path / "corrections.csv").read_bytes())) == 1
    assert json.loads((tmp_path / "corrections.json").read_text(encoding="utf-8"))["kind"] == "corrections"


@pytest.mark.parametrize("what,fmt,expected", [
    ("library", "csv", ["library.csv"]),
    ("library", "json", ["library.json"]),
    ("library", "both", ["library.csv", "library.json"]),
    ("corrections", "csv", ["corrections.csv"]),
    ("corrections", "json", ["corrections.json"]),
    ("corrections", "both", ["corrections.csv", "corrections.json"]),
    ("both", "csv", ["corrections.csv", "library.csv"]),
    ("both", "json", ["corrections.json", "library.json"]),
])
def test_what_and_format_choose_the_files(rows, tmp_path, what, fmt, expected):
    out = tmp_path / "o"
    written = X.export_files(rows, out, what, fmt)
    assert sorted(p.name for p, _ in written) == expected
    assert sorted(p.name for p in out.iterdir()) == expected


def test_unknown_what_or_format_is_refused_before_anything_is_written(rows, tmp_path):
    with pytest.raises(X.ExportError, match="--what"):
        X.export_files(rows, tmp_path / "o", "everything", "csv")
    with pytest.raises(X.ExportError, match="--format"):
        X.export_files(rows, tmp_path / "o", "library", "xml")
    assert not (tmp_path / "o").exists()


def test_the_folder_is_created_with_its_parents(rows, tmp_path):
    out = tmp_path / "a" / "b" / "c"
    X.export_files(rows, out, "library", "csv")
    assert (out / "library.csv").is_file()


def test_exporting_again_replaces_the_files(con, tmp_path):
    X.export_files(D.load_library(con), tmp_path, "corrections", "csv")
    D.approve_review(con, "3")
    X.export_files(D.load_library(con), tmp_path, "corrections", "csv")
    assert len(read_csv((tmp_path / "corrections.csv").read_bytes())) == 1


def test_a_folder_that_is_really_a_file_gives_a_clear_error(rows, tmp_path):
    blocker = tmp_path / "exports"
    blocker.write_text("i am a file")
    with pytest.raises(X.ExportError, match="cannot create the folder"):
        X.export_files(rows, blocker, "library", "csv")


def test_a_failed_write_gives_a_clear_error(rows, tmp_path, monkeypatch):
    monkeypatch.setattr(X.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("denied")))
    with pytest.raises(X.ExportError, match="cannot write into"):
        X.export_files(rows, tmp_path / "o", "library", "csv")
    assert list((tmp_path / "o").iterdir()) == []


def test_exported_at_defaults_to_now(rows, tmp_path):
    X.export_files(rows, tmp_path, "library", "json")
    stamp = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))["exported_at"]
    import re
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", stamp)


def test_a_filtered_list_exports_only_those_posts(rows, tmp_path):
    some = [r for r in rows if r["topic_id"] == "careers"]
    X.export_files(some, tmp_path, "library", "csv")
    assert [b["activity_id"] for b in read_csv((tmp_path / "library.csv").read_bytes())] == ["3"]


# ============================================================ the command line

def run_main(args, env=None):
    return X.main(args, env if env is not None else {})


def test_main_exports_and_reports(db, tmp_path, capsys):
    out = tmp_path / "out"
    assert run_main(["--db", str(db), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "library.csv  (6 rows)" in text and "corrections.json  (0 rows)" in text
    assert "6 posts in the library, 0 reviewed" in text
    assert len(read_csv((out / "library.csv").read_bytes())) == 6


def test_main_counts_reviews(db, tmp_path, capsys, con):
    D.approve_review(con, "3")
    assert run_main(["--db", str(db), "--out-dir", str(tmp_path / "o"), "--what", "corrections", "--format", "csv"]) == 0
    text = capsys.readouterr().out
    assert "corrections.csv  (1 rows)" in text and "library" not in text.split("\n")[0] and "1 reviewed" in text
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["corrections.csv"]


def test_main_takes_the_database_from_the_environment(db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # an empty folder: no data/library.sqlite to fall back on
    assert run_main(["--out-dir", str(tmp_path / "o"), "--what", "library", "--format", "json"], {"LISC_DB": str(db)}) == 0
    assert (tmp_path / "o" / "library.json").is_file()


def test_main_command_line_beats_the_environment(db, tmp_path, capsys):
    assert run_main(["--db", str(db), "--out-dir", str(tmp_path / "o")], {"LISC_DB": str(tmp_path / "nothing.sqlite")}) == 0


def test_main_default_database_and_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert run_main([]) == 2                                   # data/library.sqlite does not exist here
    assert "data/library.sqlite" in capsys.readouterr().err.replace("\\", "/")
    assert not (tmp_path / "exports").exists()


def test_main_missing_database(tmp_path, capsys):
    assert run_main(["--db", str(tmp_path / "nothing.sqlite"), "--out-dir", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and "nothing.sqlite" in err
    assert not (tmp_path / "o").exists()


def test_main_unwritable_folder(db, tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert run_main(["--db", str(db), "--out-dir", str(blocker)]) == 2
    assert capsys.readouterr().err.startswith("ERROR: cannot create the folder")


def test_main_refuses_unknown_choices(db):
    with pytest.raises(SystemExit) as exc:
        run_main(["--db", str(db), "--what", "everything"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_main(["--db", str(db), "--format", "xml"])
    assert exc.value.code == 2


def test_main_defaults_are_both_and_both(db, tmp_path):
    os.chdir(tmp_path)
    try:
        assert run_main(["--db", str(db)]) == 0
    finally:
        os.chdir(ROOT)
    assert sorted(p.name for p in (tmp_path / "exports").iterdir()) == ["corrections.csv", "corrections.json", "library.csv", "library.json"]


def test_main_ignores_an_empty_environment_value(db, tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    import shutil
    shutil.copy(db, tmp_path / "data" / "library.sqlite")
    monkeypatch.chdir(tmp_path)
    assert run_main(["--out-dir", "o"], {"LISC_DB": ""}) == 0              # falls back to data/library.sqlite
    assert (tmp_path / "o" / "library.csv").is_file()


def test_the_defaults(db):
    assert X.DEFAULT_OUT_DIR == "exports" and X.WHATS == ("library", "corrections", "both") and X.FORMATS == ("csv", "json", "both")
    assert X.LIST_SEPARATOR == " | "


# ============================================================ the dashboard's download buttons

STAMP = "2026-11-15T10:00:00+00:00"


def test_library_download_specs(rows):
    csv_spec, json_spec = X.library_downloads(rows, STAMP)
    recs = X.library_records(rows)
    assert csv_spec == {"key": "lib_dl_csv", "label": "⬇ These 6 posts as CSV", "data": X.to_csv_bytes(recs, X.LIBRARY_COLUMNS),
                        "file_name": "library.csv", "mime": "text/csv", "disabled": False}
    assert json_spec == {"key": "lib_dl_json", "label": "⬇ These 6 posts as JSON",
                         "data": X.to_json_bytes(recs, "library", STAMP),
                         "file_name": "library.json", "mime": "application/json", "disabled": False}


def test_library_download_specs_follow_the_posts_given(rows):
    one = [r for r in rows if r["activity_id"] == 3]
    csv_spec, json_spec = X.library_downloads(one, STAMP)
    assert csv_spec["label"] == "⬇ These 1 posts as CSV" and json_spec["label"] == "⬇ These 1 posts as JSON"
    assert [b["activity_id"] for b in read_csv(csv_spec["data"])] == ["3"]
    assert json.loads(json_spec["data"])["count"] == 1


def test_library_downloads_are_disabled_when_nothing_is_shown():
    csv_spec, json_spec = X.library_downloads([], STAMP)
    assert csv_spec["disabled"] is True and json_spec["disabled"] is True
    assert csv_spec["label"] == "⬇ These 0 posts as CSV" and read_csv(csv_spec["data"]) == []


def test_correction_download_specs(con):
    D.approve_review(con, "3")
    D.save_fix(con, "6", {"difficulty": "advanced"})
    rows = D.load_library(con)
    csv_spec, json_spec = X.correction_downloads(rows, STAMP)
    recs = X.correction_records(rows)
    assert csv_spec == {"key": "rv_dl_csv", "label": "⬇ My 2 reviews as CSV", "data": X.to_csv_bytes(recs, X.CORRECTION_COLUMNS),
                        "file_name": "corrections.csv", "mime": "text/csv", "disabled": False}
    assert json_spec == {"key": "rv_dl_json", "label": "⬇ My 2 reviews as JSON",
                         "data": X.to_json_bytes(recs, "corrections", STAMP),
                         "file_name": "corrections.json", "mime": "application/json", "disabled": False}


def test_correction_downloads_are_disabled_without_reviews(rows):
    csv_spec, json_spec = X.correction_downloads(rows, STAMP)
    assert csv_spec["disabled"] is True and json_spec["disabled"] is True and csv_spec["label"] == "⬇ My 0 reviews as CSV"


def test_download_time_defaults_to_now(rows):
    import re
    stamp = json.loads(X.library_downloads(rows)[1]["data"])["exported_at"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", stamp)
    assert json.loads(X.correction_downloads(rows)[1]["data"])["exported_at"] != ""


def test_a_download_matches_the_file_the_script_writes(rows, tmp_path):
    X.export_files(rows, tmp_path / "o", exported_at=STAMP)
    csv_spec, json_spec = X.library_downloads(rows, STAMP)
    assert (tmp_path / "o" / "library.csv").read_bytes() == csv_spec["data"]
    assert (tmp_path / "o" / "library.json").read_bytes() == json_spec["data"]
