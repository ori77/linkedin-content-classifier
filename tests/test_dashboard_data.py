"""Tests for pipeline/dashboard_data.py (Step 16): everything the dashboard decides, without a browser.

Uses the made-up database of tests/dash_fixture.py (built with the real loader).
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))

import dash_fixture as F  # noqa: E402
import dashboard_data as D  # noqa: E402


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


def ids(rows):
    return [r["activity_id"] for r in rows]


def by_id(rows, aid):
    return next(r for r in rows if r["activity_id"] == aid)


# ============================================================ opening the database

def test_resolve_db_path_prefers_the_command_line():
    assert D.resolve_db_path(["--db", "a.sqlite"], {"LISC_DB": "b.sqlite"}) == "a.sqlite"
    assert D.resolve_db_path(["--db=a.sqlite"], {"LISC_DB": "b.sqlite"}) == "a.sqlite"


def test_resolve_db_path_then_the_environment_then_the_default():
    assert D.resolve_db_path([], {"LISC_DB": "b.sqlite"}) == "b.sqlite"
    assert D.resolve_db_path([], {}) == D.DEFAULT_DB == "data/library.sqlite"
    assert D.resolve_db_path([], {"LISC_DB": ""}) == D.DEFAULT_DB
    assert D.resolve_db_path(["--db"], {}) == D.DEFAULT_DB            # a --db with no value is ignored
    assert D.resolve_db_path(["--other", "x"], {}) == D.DEFAULT_DB


def test_a_missing_database_says_how_to_build_it(tmp_path):
    with pytest.raises(D.DashboardError, match=r"No database at .*step15_load_sqlite\.py"):
        D.connect(tmp_path / "nope.sqlite")


def test_a_file_that_is_not_ours_is_refused(tmp_path):
    other = tmp_path / "other.sqlite"
    c = sqlite3.connect(other)
    c.execute("CREATE TABLE x(y)")
    c.commit()
    c.close()
    with pytest.raises(D.DashboardError, match="was not made by step15_load_sqlite.py"):
        D.connect(other)


def test_a_file_that_is_not_a_database_is_refused(tmp_path):
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not sqlite" * 50)
    with pytest.raises(D.DashboardError, match="not a usable database"):
        D.connect(bad)


def test_foreign_keys_are_on(con):
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ============================================================ the library rows

def test_the_library_has_the_classified_posts_in_order(rows):
    assert ids(rows) == [1, 2, 3, 4, 5, 6]


def test_every_row_has_a_post_link_and_the_resource_links(rows):
    assert all(r["post_url"] == F.url_of(r["activity_id"]) for r in rows)
    one = by_id(rows, 1)
    assert one["resource_link"] == "https://lnkd.in/a1" and one["resource_link_count"] == 3
    assert one["resource_links_list"] == ["https://lnkd.in/a1", "https://example.com/two", "https://lnkd.in/a3"]
    none = by_id(rows, 2)
    assert none["resource_link"] is None and none["resource_links_list"] == []


def test_row_fields(rows):
    r = by_id(rows, 3)
    assert r["title"] == "Interview prep" and r["author"] == "Cy" and r["saved_at"] == "2026-09-01 10:00:00"
    assert r["topic_name"] == "Career Development" and r["subtopic_name"] == "Interviews"
    assert (r["cover_minutes"], r["understand_minutes"]) == (40, 40)
    assert r["confidence"] == 0.61 and r["review_reasons"] == ["low_confidence", "topic_disagrees_with_clustering"]
    assert r["model_flagged"] is True and r["reviewed"] is False and r["status"] == "needs_review"
    assert r["corrected_fields"] == {} and r["review_note"] is None
    assert (r["model_subtopic_id"], r["model_difficulty"], r["model_content_type"]) == (
        "careers.interviews", "intermediate", "interview_question_set")
    assert r["taxonomy_version"] == "v1" and r["classification_id"].startswith("cls_")


def test_status_of_unreviewed_rows(rows):
    assert {r["activity_id"]: r["status"] for r in rows} == {
        1: "auto_accepted", 2: "auto_accepted", 3: "needs_review", 4: "needs_review", 5: "needs_review", 6: "needs_review"}


def test_the_post_without_text_is_not_in_the_library(rows):
    assert 7 not in ids(rows)


def test_rows_use_the_loaders_recomputed_flags(rows):
    assert by_id(rows, 5)["review_reasons"] == ["other_general"] and by_id(rows, 5)["confidence"] == 0.93


# ============================================================ corrections laid over the model's labels

def test_a_fix_changes_what_the_library_shows_but_keeps_the_model_labels(con):
    D.save_fix(con, "3", {"subtopic_id": "rag.chunking", "difficulty": "advanced", "content_type": "tutorial"})
    r = by_id(D.load_library(con), 3)
    assert (r["subtopic_id"], r["topic_id"], r["difficulty"], r["content_type"]) == ("rag.chunking", "rag", "advanced", "tutorial")
    assert (r["topic_name"], r["subtopic_name"]) == ("Retrieval-Augmented Generation", "Chunking")
    assert (r["model_subtopic_id"], r["model_difficulty"], r["model_content_type"]) == (
        "careers.interviews", "intermediate", "interview_question_set")
    assert r["reviewed"] is True and r["status"] == "reviewed"
    assert r["corrected_fields"] == {"subtopic_id": "rag.chunking", "difficulty": "advanced", "content_type": "tutorial"}


def test_an_approval_leaves_the_labels_and_marks_the_post_reviewed(con):
    D.approve_review(con, "3", "looks right")
    r = by_id(D.load_library(con), 3)
    assert r["subtopic_id"] == "careers.interviews" and r["difficulty"] == "intermediate"
    assert r["reviewed"] is True and r["status"] == "reviewed" and r["corrected_fields"] == {}
    assert r["review_note"] == "looks right"


def test_the_newest_review_wins(con):
    D.save_fix(con, "3", {"difficulty": "advanced"})
    D.save_fix(con, "3", {"difficulty": "beginner"})
    assert by_id(D.load_library(con), 3)["difficulty"] == "beginner"
    D.approve_review(con, "3")                                   # "the model was right" overrides the earlier fixes
    r = by_id(D.load_library(con), 3)
    assert r["difficulty"] == "intermediate" and r["corrected_fields"] == {}


def test_the_review_with_the_later_time_wins_whatever_order_it_was_written_in(con):
    with con:                                    # the NEWER review is written first, so only its time says it is newer
        for rid, stamp, diff in (("r1", "2026-11-14T10:00:00+00:00", "advanced"), ("r2", "2026-11-14T09:00:00+00:00", "beginner")):
            con.execute("INSERT INTO reviews(id, item_id, classification_id, corrected_fields, created_at) VALUES (?, '3', NULL, ?, ?)",
                        (rid, json.dumps({"difficulty": diff}), stamp))
    assert by_id(D.load_library(con), 3)["difficulty"] == "advanced"


def test_other_posts_are_not_touched_by_a_review(con):
    before = by_id(D.load_library(con), 4)
    D.save_fix(con, "3", {"difficulty": "advanced"})
    assert by_id(D.load_library(con), 4) == before


def test_a_correction_with_an_unknown_value_or_field_is_ignored_when_reading(con):
    for corrected in ({"difficulty": "godlike"}, {"content_type": "poem"}, {"subtopic_id": "nope.nope"},
                      {"colour": "red"}, ["not", "a", "dict"]):
        with con:
            con.execute("DELETE FROM reviews")
            con.execute("INSERT INTO reviews(id, item_id, classification_id, corrected_fields, created_at) "
                        "VALUES ('r', '3', NULL, ?, 't')", (json.dumps(corrected),))
        r = by_id(D.load_library(con), 3)
        assert r["difficulty"] == "intermediate" and r["subtopic_id"] == "careers.interviews"
        assert r["content_type"] == "interview_question_set" and r["corrected_fields"] == {}
        assert r["reviewed"] is True                              # the review exists, so the post counts as reviewed


def test_a_review_with_broken_json_does_not_crash(con):
    with con:
        con.execute("INSERT INTO reviews(id, item_id, corrected_fields, created_at) VALUES ('r', '3', '{}', 't')")
        con.execute("PRAGMA ignore_check_constraints = ON")
        con.execute("UPDATE reviews SET corrected_fields = 'not json' WHERE id = 'r'")
    assert by_id(D.load_library(con), 3)["reviewed"] is True


def test_the_fix_survives_a_reload_of_the_database(db, tmp_path):
    c = D.connect(db)
    D.save_fix(c, "3", {"difficulty": "advanced"})
    c.close()
    import step15_load_sqlite as L
    L.load(db, tmp_path / "run1.jsonl", tmp_path / "posts.csv", tmp_path / "tax.yaml", tmp_path / "rule.yaml")
    c = D.connect(db)
    try:
        assert by_id(D.load_library(c), 3)["difficulty"] == "advanced"
    finally:
        c.close()


# ============================================================ saving decisions

def test_approve_stores_an_empty_set_and_the_active_classification(con):
    rid = D.approve_review(con, "3", "  fine  ", use_as_example=False)
    row = con.execute("SELECT * FROM reviews WHERE id = ?", (rid,)).fetchone()
    active = con.execute("SELECT id FROM classifications WHERE item_id = '3' AND is_active = 1").fetchone()[0]
    assert rid.startswith("rev_") and row["item_id"] == "3" and row["classification_id"] == active
    assert row["corrected_fields"] == "{}" and row["note"] == "fine" and row["use_as_example"] == 0
    assert row["created_at"].startswith("20")


def test_an_empty_note_is_stored_as_nothing(con):
    rid = D.approve_review(con, "3", "   ")
    assert con.execute("SELECT note FROM reviews WHERE id = ?", (rid,)).fetchone()[0] is None


def test_use_as_example_defaults_to_yes(con):
    rid = D.approve_review(con, "3")
    assert con.execute("SELECT use_as_example FROM reviews WHERE id = ?", (rid,)).fetchone()[0] == 1


def test_save_fix_stores_only_what_differs_from_the_model(con):
    rid, saved = D.save_fix(con, "3", {"subtopic_id": "careers.interviews", "difficulty": "advanced",
                                       "content_type": "interview_question_set"}, "harder than it looks")
    assert saved == {"difficulty": "advanced"}
    row = con.execute("SELECT * FROM reviews WHERE id = ?", (rid,)).fetchone()
    assert json.loads(row["corrected_fields"]) == {"difficulty": "advanced"} and row["note"] == "harder than it looks"


def test_save_fix_compares_with_the_model_not_with_an_earlier_fix(con):
    D.save_fix(con, "3", {"difficulty": "advanced"})
    _, saved = D.save_fix(con, "3", {"difficulty": "advanced", "content_type": "tutorial"})
    assert saved == {"difficulty": "advanced", "content_type": "tutorial"}      # difficulty is still a correction


def test_save_fix_with_nothing_changed_is_refused_and_saves_nothing(con):
    with pytest.raises(D.DashboardError, match="Nothing differs from the model's labels"):
        D.save_fix(con, "3", {"subtopic_id": "careers.interviews", "difficulty": "intermediate"})
    assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_save_fix_with_an_empty_choice_is_refused(con):
    with pytest.raises(D.DashboardError, match="Nothing differs"):
        D.save_fix(con, "3", {})


def test_save_fix_rejects_values_that_are_not_allowed(con):
    for key, value in (("difficulty", "godlike"), ("content_type", "poem"), ("subtopic_id", "rag.unknown")):
        with pytest.raises(D.DashboardError, match=f"'{value}' is not a valid {key}"):
            D.save_fix(con, "3", {key: value})
    assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_save_fix_rejects_labels_that_cannot_be_corrected(con):
    with pytest.raises(D.DashboardError, match="cannot correct"):
        D.save_fix(con, "3", {"summary": "new text"})


def test_other_general_is_a_valid_correction(con):
    _, saved = D.save_fix(con, "3", {"subtopic_id": "other.general"})
    assert saved == {"subtopic_id": "other.general"}
    r = by_id(D.load_library(con), 3)
    assert (r["topic_id"], r["topic_name"], r["subtopic_name"]) == ("other", "Other / General", "Other / General")


def test_a_post_that_does_not_exist_cannot_be_reviewed(con):
    for func in (lambda: D.approve_review(con, "999"), lambda: D.save_fix(con, "999", {"difficulty": "advanced"}),
                 lambda: D.approve_review(con, "7")):                       # 7 has a post link but no classification
        with pytest.raises(D.DashboardError, match="has no active classification"):
            func()


def test_a_review_points_at_the_classification_that_was_active(con):
    rid = D.approve_review(con, "3")
    cls = con.execute("SELECT classification_id FROM reviews WHERE id = ?", (rid,)).fetchone()[0]
    assert con.execute("SELECT item_id FROM classifications WHERE id = ?", (cls,)).fetchone()[0] == "3"


def test_two_reviews_in_the_same_second_keep_their_order(con, monkeypatch):
    monkeypatch.setattr(D, "utc_now", lambda: "2026-11-14T10:00:00+00:00")
    D.save_fix(con, "3", {"difficulty": "advanced"})
    D.save_fix(con, "3", {"difficulty": "beginner"})
    assert by_id(D.load_library(con), 3)["difficulty"] == "beginner"


# ============================================================ filters

def f(**kw):
    return D.Filters(**kw)


def test_no_filter_keeps_everything(rows):
    assert D.apply_filters(rows, f()) == rows


def test_filter_by_topic_and_subtopic(rows):
    assert ids(D.apply_filters(rows, f(topics=("rag",)))) == [1, 2, 4, 6]
    assert ids(D.apply_filters(rows, f(topics=("careers", "other")))) == [3, 5]
    assert ids(D.apply_filters(rows, f(subtopics=("rag.chunking",)))) == [1, 4]
    assert ids(D.apply_filters(rows, f(topics=("rag",), subtopics=("careers.interviews",)))) == []


def test_filter_by_difficulty_and_type(rows):
    assert ids(D.apply_filters(rows, f(difficulties=("beginner",)))) == [1, 4, 5]
    assert ids(D.apply_filters(rows, f(difficulties=("beginner", "advanced")))) == [1, 2, 4, 5]
    assert ids(D.apply_filters(rows, f(content_types=("tutorial",)))) == [1, 6]
    assert ids(D.apply_filters(rows, f(difficulties=("beginner",), content_types=("tutorial",)))) == [1]


def test_filter_by_alignment_score_range_includes_the_edges(rows):
    assert ids(D.apply_filters(rows, f(min_score=5.0, max_score=7.0))) == [2, 4, 6]
    assert ids(D.apply_filters(rows, f(min_score=8.0))) == [1]
    assert ids(D.apply_filters(rows, f(max_score=1.0))) == [5]
    assert ids(D.apply_filters(rows, f(min_score=10.0))) == [1]
    assert ids(D.apply_filters(rows, f(max_score=0.5))) == []


def test_filter_by_alignment_band(rows):
    assert ids(D.apply_filters(rows, f(bands=("Must-read",)))) == [1]
    assert ids(D.apply_filters(rows, f(bands=("Low relevance", "Highly relevant")))) == [2, 3, 5, 6]


def test_filter_by_time_bucket(rows):
    assert ids(D.apply_filters(rows, f(cover_buckets=("<15m",)))) == [4, 5]
    assert ids(D.apply_filters(rows, f(understand_buckets=("1-3d",)))) == [2]
    assert ids(D.apply_filters(rows, f(cover_buckets=("3-8h",), understand_buckets=("3-8h",)))) == [6]
    assert ids(D.apply_filters(rows, f(cover_buckets=("3-8h",), understand_buckets=("1-3h",)))) == []


def test_filter_by_confidence(rows):
    assert ids(D.apply_filters(rows, f(min_confidence=0.9))) == [1, 2, 5]
    assert ids(D.apply_filters(rows, f(min_confidence=0.93))) == [1, 2, 5]
    assert ids(D.apply_filters(rows, f(min_confidence=0.94))) == [1, 2]


def test_filter_by_review_status(con):
    D.approve_review(con, "3")
    rows = D.load_library(con)
    assert ids(D.apply_filters(rows, f(status="needs_review"))) == [4, 5, 6]
    assert ids(D.apply_filters(rows, f(status="reviewed"))) == [3]
    assert ids(D.apply_filters(rows, f(status="auto_accepted"))) == [1, 2]
    assert ids(D.apply_filters(rows, f(status="all"))) == [1, 2, 3, 4, 5, 6]


def test_an_unknown_status_is_an_error(rows):
    with pytest.raises(D.DashboardError, match="unknown status filter"):
        D.apply_filters(rows, f(status="whatever"))


def test_the_filters_use_the_corrected_labels(con):
    D.save_fix(con, "3", {"difficulty": "advanced", "subtopic_id": "rag.chunking"})
    rows = D.load_library(con)
    assert 3 in ids(D.apply_filters(rows, f(difficulties=("advanced",))))
    assert 3 not in ids(D.apply_filters(rows, f(difficulties=("intermediate",))))
    assert 3 in ids(D.apply_filters(rows, f(topics=("rag",))))
    assert 3 not in ids(D.apply_filters(rows, f(topics=("careers",))))


def test_text_search_looks_in_title_summary_author_and_names(rows):
    assert ids(D.apply_filters(rows, f(text="chunking"))) == [1, 4]            # title of 1; subtopic name of 1 and 4
    assert ids(D.apply_filters(rows, f(text="summary of post 6"))) == [6]
    assert ids(D.apply_filters(rows, f(text="fay"))) == [6]
    assert ids(D.apply_filters(rows, f(text="career development"))) == [3]
    assert ids(D.apply_filters(rows, f(text="  CHEAP   tricks "))) == [4]       # case, spaces and word order do not matter
    assert ids(D.apply_filters(rows, f(text="tricks cheap"))) == [4]
    assert ids(D.apply_filters(rows, f(text="cheap nothing"))) == []            # every word must be found
    assert ids(D.apply_filters(rows, f(text="   "))) == ids(rows)


def test_text_search_does_not_look_in_the_post_text(rows):
    assert ids(D.apply_filters(rows, f(text="body text"))) == []


def test_filters_combine(rows):
    got = D.apply_filters(rows, f(topics=("rag",), difficulties=("beginner",), min_score=5.0, status="needs_review"))
    assert ids(got) == [4]


# ============================================================ sorting

def test_default_sort_is_highest_alignment_first(rows):
    assert ids(D.sort_rows(rows)) == [1, 6, 2, 4, 3, 5]
    assert D.DEFAULT_SORT in D.SORTS


def test_sort_by_confidence_lowest_first(rows):
    assert ids(D.sort_rows(rows, "Confidence (low first)")) == [3, 6, 4, 5, 2, 1]


def test_sort_by_times(rows):
    assert ids(D.sort_rows(rows, "Cover time (short first)")) == [4, 5, 1, 3, 6, 2]
    assert ids(D.sort_rows(rows, "Understand time (short first)")) == [4, 5, 3, 1, 6, 2]


def test_sort_by_saved_date_puts_the_undated_post_last(rows):
    assert ids(D.sort_rows(rows, "Saved date (newest first)")) == [2, 1, 3, 4, 6, 5]


def test_sort_by_title(rows):
    assert ids(D.sort_rows(rows, "Title (A to Z)")) == [4, 1, 2, 3, 5, 6]


def test_sort_ties_are_broken_by_post_id(rows):
    tied = [dict(r, alignment_score=5.0) for r in reversed(rows)]
    assert ids(D.sort_rows(tied)) == [1, 2, 3, 4, 5, 6]


def test_sort_does_not_change_the_input(rows):
    before = ids(rows)
    D.sort_rows(rows, "Title (A to Z)")
    assert ids(rows) == before


def test_unknown_sort_is_an_error(rows):
    with pytest.raises(D.DashboardError, match="unknown sort"):
        D.sort_rows(rows, "by colour")


# ============================================================ what the filter widgets offer

def test_filter_options(rows):
    o = D.filter_options(rows)
    assert o["topics"] == [("careers", "Career Development"), ("other", "Other / General"),
                           ("rag", "Retrieval-Augmented Generation")]
    assert o["subtopics"] == [("rag.chunking", "Chunking", "rag"), ("careers.interviews", "Interviews", "careers"),
                              ("other.general", "Other / General", "other"), ("rag.evaluation", "RAG Evaluation", "rag")]
    assert o["difficulties"] == ["beginner", "intermediate", "advanced"]                  # easiest first, expert unused
    assert o["content_types"] == ["interview_question_set", "full_project", "tutorial", "miscellaneous"]
    assert o["bands"] == ["Must-read", "Highly relevant", "Useful background", "Low relevance"]   # best first
    assert o["cover_buckets"] == ["<15m", "15-60m", "3-8h"]                                # shortest first
    assert o["understand_buckets"] == ["<15m", "15-60m", "1-3h", "3-8h", "1-3d"]


def test_options_follow_the_corrected_labels(con):
    D.save_fix(con, "2", {"difficulty": "expert"})
    assert "expert" in D.filter_options(D.load_library(con))["difficulties"]


# ============================================================ counts and review queue

def test_counts(con):
    assert D.counts(D.load_library(con)) == {"total": 6, "flagged_by_model": 4, "reviewed": 0, "to_review": 4}
    D.approve_review(con, "3")
    D.save_fix(con, "1", {"difficulty": "advanced"})           # an auto-accepted post can be reviewed too
    assert D.counts(D.load_library(con)) == {"total": 6, "flagged_by_model": 4, "reviewed": 2, "to_review": 3}


def test_the_queue_is_the_flagged_posts_lowest_confidence_first(rows):
    assert ids(D.review_queue(rows)) == [3, 6, 4, 5]


def test_reviewed_posts_leave_the_queue_unless_asked(con):
    D.approve_review(con, "3")
    rows = D.load_library(con)
    assert ids(D.review_queue(rows)) == [6, 4, 5]
    assert ids(D.review_queue(rows, include_reviewed=True)) == [3, 6, 4, 5]


def test_unflagged_posts_are_not_in_the_queue_even_when_reviewed(con):
    D.approve_review(con, "1")
    assert 1 not in ids(D.review_queue(D.load_library(con), include_reviewed=True))


def test_queue_ties_go_by_post_id(rows):
    tied = [dict(r, confidence=0.5) for r in reversed(rows)]
    assert ids(D.review_queue(tied)) == [3, 4, 5, 6]


def test_subtopic_choices(con):
    assert D.subtopic_choices(con, "v1") == [
        ("careers.interviews", "Career Development > Interviews"),
        ("rag.chunking", "Retrieval-Augmented Generation > Chunking"),
        ("rag.evaluation", "Retrieval-Augmented Generation > RAG Evaluation"),
        ("other.general", "Other / General")]
    assert D.subtopic_choices(con, "v9") == []


def test_review_detail(con, rows):
    d = D.review_detail(con, by_id(rows, 3)["classification_id"])
    assert d["body_text"] == "Interview prep\nbody text of post 3" and d["taxonomy_version"] == "v1"
    assert d["rationales"]["topic"] == "topic reason 3" and d["key_points"] == ["first point", "second point"]
    assert d["warnings"] == ["interview_mentioned"] and d["attempts"] == {"topic": 1, "rest": 1}
    assert d["time"]["basis"] == "post_only" and d["alignment_subscores"] == {"applicability": 7}
    assert d["confidence_parts"]["agree"] == 0.4 and d["review_reasons"] == ["low_confidence", "topic_disagrees_with_clustering"]
    assert d["clustering_subtopic_id"] == "rag.chunking" and d["clustering_name"] == "Chunking"
    assert d["confidence_as_run"] == 0.5 and d["model"] == "gemma4:e4b"


def test_review_detail_for_a_post_clustering_could_not_place(con, rows):
    d = D.review_detail(con, by_id(rows, 5)["classification_id"])
    assert d["clustering_subtopic_id"] == "other.general" and d["clustering_name"] == "Other / General"


def test_review_detail_of_something_that_does_not_exist(con):
    with pytest.raises(D.DashboardError, match="no classification"):
        D.review_detail(con, "cls_nope")


# ============================================================ showing things

def test_format_minutes():
    assert D.format_minutes(None) == "?"
    assert D.format_minutes(5) == "5m" and D.format_minutes(59) == "59m"
    assert D.format_minutes(60) == "1h" and D.format_minutes(90) == "1h 30m"
    assert D.format_minutes(1439) == "23h 59m"
    assert D.format_minutes(1440) == "1d" and D.format_minutes(1500) == "1d 1h" and D.format_minutes(3600) == "2d 12h"
    assert D.format_minutes(89.6) == "1h 30m"


def test_md_escape_stops_accidental_formatting():
    assert D.md_escape(None) == "" and D.md_escape("plain words") == "plain words"
    assert D.md_escape("*bold* _it_ `code`") == r"\*bold\* \_it\_ \`code\`"
    assert D.md_escape("$5 and $10") == r"\$5 and \$10"
    assert D.md_escape("[a](b)") == r"\[a\]\(b\)"
    assert D.md_escape("<b>x</b> # y | z") == r"\<b\>x\</b\> \# y \| z"
    assert D.md_escape("1. one\n- two\n  3. three") == "1\\. one\n\\- two\n  3\\. three"
    assert D.md_escape("a-b and 1.5") == "a-b and 1.5"
    assert D.md_escape("a\r\nb") == "a\nb"
    assert D.md_escape("back\\slash") == "back\\\\slash"


def test_md_link_list():
    assert D.md_link_list([]) == ""
    out = D.md_link_list(["https://lnkd.in/a", "https://x.com/f(2)", "https://x.com/a b"])
    assert out.splitlines() == [r"- [https\://lnkd.in/a](https://lnkd.in/a)",
                                r"- [https\://x.com/f\(2\)](https://x.com/f%282%29)",
                                r"- [https\://x.com/a b](https://x.com/a%20b)"]


def test_table_rows_have_both_links_and_readable_values(rows):
    t = D.table_rows([by_id(rows, 1), by_id(rows, 2)])
    assert list(t[0]) == ["Title", "Author", "Topic", "Subtopic", "Difficulty", "Type", "Cover", "Understand",
                          "AI-engineer score", "Band", "Confidence", "Status", "Summary", "Post", "Resource"]
    assert t[0]["Post"] == F.url_of(1) and t[0]["Resource"] == "https://lnkd.in/a1"
    assert t[1]["Post"] == F.url_of(2) and t[1]["Resource"] is None
    assert t[0]["Cover"] == "30m" and t[0]["Understand"] == "1h 30m" and t[1]["Understand"] == "1d 1h"
    assert t[0]["AI-engineer score"] == 10.0 and t[0]["Confidence"] == 0.98
    assert t[0]["Status"] == "auto-accepted" and t[0]["Topic"] == "Retrieval-Augmented Generation"


def test_table_rows_other_columns(rows):
    t = D.table_rows([by_id(rows, 3)])[0]
    assert (t["Title"], t["Author"], t["Subtopic"], t["Difficulty"], t["Type"]) == (
        "Interview prep", "Cy", "Interviews", "intermediate", "interview_question_set")
    assert (t["Band"], t["Summary"], t["Cover"], t["Understand"]) == ("Low relevance", "Summary of post 3.", "40m", "40m")


def test_table_rows_status_words(con):
    D.approve_review(con, "3")
    t = {r["Title"]: r["Status"] for r in D.table_rows(D.load_library(con))}
    assert t["Interview prep"] == "reviewed" and t["Cheap tricks"] == "needs review" and t["Chunking guide"] == "auto-accepted"


def test_table_rows_without_title_or_author(rows):
    r = dict(by_id(rows, 1), title=None, author=None)
    t = D.table_rows([r])[0]
    assert t["Title"] == "(no title)" and t["Author"] == ""


def test_the_editable_labels_and_choices():
    assert D.EDITABLE == ("subtopic_id", "difficulty", "content_type")
    assert D.STATUS_CHOICES == ("all", "needs_review", "reviewed", "auto_accepted")
    assert set(D.STATUS_LABELS) == set(D.STATUS_CHOICES)


# ============================================================ pages of cards

def test_page_count():
    assert D.page_count(0, 20) == 1 and D.page_count(1, 20) == 1 and D.page_count(20, 20) == 1
    assert D.page_count(21, 20) == 2 and D.page_count(40, 20) == 2 and D.page_count(41, 20) == 3


def test_page_of(rows):
    assert ids(D.page_of(rows, 1, 4)) == [1, 2, 3, 4]
    assert ids(D.page_of(rows, 2, 4)) == [5, 6]
    assert ids(D.page_of(rows, 3, 4)) == [5, 6]            # past the end -> the last page
    assert ids(D.page_of(rows, 0, 4)) == [1, 2, 3, 4]      # before the start -> the first page
    assert ids(D.page_of(rows, -5, 4)) == [1, 2, 3, 4]
    assert D.page_of([], 1, 4) == []
    assert ids(D.page_of(rows, 1, 20)) == ids(rows)


# ============================================================ more edge cases

SPECIALS = "\\`*_{}[]()#+!|<>~$&:"


@pytest.mark.parametrize("ch", list(SPECIALS))
def test_md_escape_escapes_every_special_character(ch):
    assert D.md_escape(f"a{ch}b") == f"a\\{ch}b"


def test_the_title_sort_ignores_capital_letters(con):
    with con:
        con.execute("UPDATE items SET title = 'apple' WHERE id = '2'")
    rows = D.load_library(con)
    assert ids(D.sort_rows(rows, "Title (A to Z)"))[:2] == [2, 4]            # apple, Cheap tricks


def test_option_names_are_sorted_ignoring_capital_letters(rows):
    renamed = [dict(r, topic_name="a topic") if r["topic_id"] == "rag" else dict(r, topic_name="B topic") for r in rows]
    names = [n for _, n in D.filter_options(renamed)["topics"]]
    assert names[0] == "a topic" and names[-1] == "B topic"
    subs = [dict(r, subtopic_name="a sub") if r["subtopic_id"] == "rag.chunking" else dict(r, subtopic_name="B sub") for r in rows]
    names = [n for _, n, _ in D.filter_options(subs)["subtopics"]]
    assert names[0] == "a sub" and names[-1] == "B sub"


def test_a_review_goes_to_the_newest_classification(db, tmp_path, con):
    import step15_load_sqlite as L
    newer = F.make_record(3, difficulty="advanced", cache_key="newer-3" + "0" * 40, created_at="2026-11-20T03:00:00+00:00")
    (tmp_path / "run2.jsonl").write_text(json.dumps(newer) + "\n", encoding="utf-8")
    L.load(db, tmp_path / "run2.jsonl", tmp_path / "posts.csv", tmp_path / "tax.yaml", tmp_path / "rule.yaml")
    with pytest.raises(D.DashboardError, match="Nothing differs"):               # compared with the NEW labels
        D.save_fix(con, "3", {"difficulty": "advanced"})
    rid = D.approve_review(con, "3")
    new_id = con.execute("SELECT id FROM classifications WHERE item_id = '3' AND is_active = 1").fetchone()[0]
    assert con.execute("SELECT classification_id FROM reviews WHERE id = ?", (rid,)).fetchone()[0] == new_id
    assert by_id(D.load_library(con), 3)["model_difficulty"] == "advanced"


def test_review_time_is_a_utc_timestamp(con):
    rid = D.approve_review(con, "3")
    stamp = con.execute("SELECT created_at FROM reviews WHERE id = ?", (rid,)).fetchone()[0]
    import re
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", stamp)


def test_pick_row(rows):
    assert D.pick_row(rows, [0]) is rows[0] and D.pick_row(rows, [5]) is rows[5]
    assert D.pick_row(rows, []) is None
    assert D.pick_row(rows, [6]) is None and D.pick_row(rows, [99]) is None
    assert D.pick_row(rows, [-1]) is None
    assert D.pick_row([], [0]) is None
