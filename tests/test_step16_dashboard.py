"""Tests for pipeline/step16_dashboard.py (Step 16): the screens, run with Streamlit's own test runner.

The decisions behind the screens are tested in test_dashboard_data.py. These tests check that the
screens use them: the right posts are shown, every post has its links, the filters change the
list, and the Review buttons save the right thing. Skipped when streamlit is not installed.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))

import dash_fixture as F  # noqa: E402

APP = str(ROOT / "pipeline" / "step16_dashboard.py")
QUEUE = [3, 6, 4, 5]              # review order: lowest confidence first


@pytest.fixture
def db(tmp_path):
    return F.build(tmp_path)


def start(db, monkeypatch, argv=None):
    monkeypatch.setenv("LISC_DB", str(db))
    monkeypatch.setattr(sys, "argv", argv or ["step16_dashboard.py"])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def reviews(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM reviews ORDER BY created_at, rowid")]
    finally:
        con.close()


def table(at):
    return at.dataframe[0].value


def links(block):
    return [(b.proto.label, b.proto.url) for b in block.get("link_button")]


def review_title(at):
    return at.tabs[1].subheader[0].value


def review_counts(at):
    return at.tabs[1].caption[0].value


# ============================================================ starting up

def test_the_page_loads_with_both_tabs(db, monkeypatch):
    at = start(db, monkeypatch)
    assert at.title[0].value == "LinkedIn saved posts"
    assert [t.label for t in at.tabs] == ["📚 Library", "🟡 Review"]
    assert "6 classified posts" in at.caption[0].value and str(db) in at.caption[0].value


def test_the_database_can_come_from_the_command_line(db, monkeypatch):
    monkeypatch.delenv("LISC_DB", raising=False)
    monkeypatch.setattr(sys, "argv", ["step16_dashboard.py", "--db", str(db)])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and "6 classified posts" in at.caption[0].value


def test_a_missing_database_shows_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LISC_DB", str(tmp_path / "nope.sqlite"))
    monkeypatch.setattr(sys, "argv", ["step16_dashboard.py"])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No database at" in at.error[0].value and "step15_load_sqlite.py" in at.error[0].value
    assert len(at.tabs) == 0


def test_a_database_without_classified_posts_says_so(db, monkeypatch):
    con = sqlite3.connect(db)
    con.execute("DELETE FROM classifications")
    con.commit()
    con.close()
    at = start(db, monkeypatch)
    assert "no classified posts" in at.warning[0].value and len(at.tabs) == 0


# ============================================================ Library: table

def test_the_table_lists_every_post_with_both_link_columns(db, monkeypatch):
    at = start(db, monkeypatch)
    t = table(at)
    assert len(t) == 6 and set(t["Title"]) == {"Chunking guide", "Evaluation project", "Interview prep",
                                                "Cheap tricks", "Odd one", "Tips & $tricks* 100%"}
    assert all(u == F.url_of(i) for u, i in zip(t["Post"], [1, 6, 2, 4, 3, 5]))        # default sort: score
    assert [u for u in t["Resource"] if isinstance(u, str)] == ["https://lnkd.in/a1", "https://lnkd.in/f1", "https://lnkd.in/c1"]
    cfg = json.loads(at.dataframe[0].proto.columns)
    assert cfg["Post"]["type_config"] == {"type": "link", "display_text": "Open post"}
    assert cfg["Resource"]["type_config"] == {"type": "link", "display_text": "Open resource"}


def test_the_table_hint_and_count(db, monkeypatch):
    at = start(db, monkeypatch)
    assert "**6 of 6 posts**" in [m.value for m in at.tabs[0].markdown]
    assert "Click a row" in at.tabs[0].caption[0].value


def test_sorting_changes_the_table(db, monkeypatch):
    at = start(db, monkeypatch)
    assert list(table(at)["Title"])[0] == "Chunking guide"
    at.selectbox(key="lib_sort").select("Title (A to Z)").run()
    assert list(table(at)["Title"])[:2] == ["Cheap tricks", "Chunking guide"]
    at.selectbox(key="lib_sort").select("Confidence (low first)").run()
    assert list(table(at)["Title"])[:2] == ["Interview prep", "Tips & $tricks* 100%"]


# ============================================================ Library: filters

def count_text(at):
    return next(m.value for m in at.tabs[0].markdown if " posts**" in m.value)


def test_topic_filter_and_the_subtopic_list_follows_it(db, monkeypatch):
    at = start(db, monkeypatch)
    at.multiselect(key="lib_topics").select("rag").run()
    assert count_text(at) == "**4 of 6 posts**" and set(table(at)["Topic"]) == {"Retrieval-Augmented Generation"}
    assert at.multiselect(key="lib_subtopics").options == ["Chunking", "RAG Evaluation"]
    at.multiselect(key="lib_subtopics").select("rag.chunking").run()
    assert count_text(at) == "**2 of 6 posts**"


def test_difficulty_and_type_filters(db, monkeypatch):
    at = start(db, monkeypatch)
    at.multiselect(key="lib_difficulties").select("beginner").run()
    assert count_text(at) == "**3 of 6 posts**"
    at.multiselect(key="lib_types").select("tutorial").run()
    assert count_text(at) == "**1 of 6 posts**" and list(table(at)["Title"]) == ["Chunking guide"]


def test_alignment_score_slider_and_band_filter(db, monkeypatch):
    at = start(db, monkeypatch)
    at.slider(key="lib_score").set_value((5.0, 7.0)).run()
    assert count_text(at) == "**3 of 6 posts**" and set(table(at)["AI-engineer score"]) == {5.0, 6.0, 7.0}
    at.slider(key="lib_score").set_value((0.0, 10.0)).run()
    at.multiselect(key="lib_bands").select("Must-read").run()
    assert count_text(at) == "**1 of 6 posts**"


def test_time_filters(db, monkeypatch):
    at = start(db, monkeypatch)
    at.multiselect(key="lib_cover").select("<15m").run()
    assert count_text(at) == "**2 of 6 posts**"
    at.multiselect(key="lib_understand").select("1-3d").run()
    assert count_text(at) == "**0 of 6 posts**"


def test_confidence_and_status_filters(db, monkeypatch):
    at = start(db, monkeypatch)
    at.slider(key="lib_conf").set_value(0.9).run()
    assert count_text(at) == "**3 of 6 posts**"
    at.slider(key="lib_conf").set_value(0.0).run()
    at.selectbox(key="lib_status").select("needs_review").run()
    assert count_text(at) == "**4 of 6 posts**"
    at.selectbox(key="lib_status").select("auto_accepted").run()
    assert count_text(at) == "**2 of 6 posts**"


def test_text_search(db, monkeypatch):
    at = start(db, monkeypatch)
    at.text_input(key="lib_text").set_value("cheap").run()
    assert list(table(at)["Title"]) == ["Cheap tricks"]


def test_no_match_shows_a_message_instead_of_an_empty_table(db, monkeypatch):
    at = start(db, monkeypatch)
    at.text_input(key="lib_text").set_value("zzzz").run()
    assert "No posts match" in at.tabs[0].info[0].value and len(at.dataframe) == 0


# ============================================================ Library: cards

def to_cards(at):
    at.radio(key="lib_view").set_value("Cards").run()
    return at


def test_cards_show_the_title_the_facts_and_the_summary(db, monkeypatch):
    at = to_cards(start(db, monkeypatch))
    text = [m.value for m in at.tabs[0].markdown]
    assert "**Chunking guide**" in text and "Summary of post 1." in text
    assert any("`beginner` · `tutorial` · cover 30m · understand 1h 30m · AI-engineer score 10.0/10 · confidence 0.98" in t for t in text)
    assert any("⚠ needs review" in c.value for c in at.tabs[0].caption)
    assert len(at.dataframe) == 0


def test_every_card_has_a_post_link_and_only_posts_with_links_have_a_resource_link(db, monkeypatch):
    at = to_cards(start(db, monkeypatch))
    got = links(at.tabs[0])
    posts = [u for label, u in got if label == "Open post"]
    resources = [u for label, u in got if label == "Open resource"]
    assert posts == [F.url_of(i) for i in (1, 6, 2, 4, 3, 5)]
    assert resources == ["https://lnkd.in/a1", "https://lnkd.in/f1", "https://lnkd.in/c1"]


def test_posts_with_several_links_list_them_all(db, monkeypatch):
    at = to_cards(start(db, monkeypatch))
    labels = [e.label for e in at.tabs[0].get("expander")]
    assert labels == ["All 3 links found in the post", "All 2 links found in the post"]       # posts 1 and 6, not 3


def test_card_text_is_escaped(db, monkeypatch):
    at = to_cards(start(db, monkeypatch))
    assert r"**Tips \& \$tricks\* 100%**" in [m.value for m in at.tabs[0].markdown]


def test_cards_follow_the_filters_and_the_sort(db, monkeypatch):
    at = start(db, monkeypatch)
    at.multiselect(key="lib_topics").select("rag").run()
    at.selectbox(key="lib_sort").select("Title (A to Z)").run()
    to_cards(at)
    assert [u for label, u in links(at.tabs[0]) if label == "Open post"] == [F.url_of(i) for i in (4, 1, 2, 6)]


# ============================================================ Review: what is shown

def test_the_review_tab_starts_with_the_lowest_confidence_post(db, monkeypatch):
    at = start(db, monkeypatch)
    assert review_title(at) == "Interview prep"
    assert review_counts(at) == "4 posts flagged by the model · 0 reviewed · 4 left in this queue"


def test_the_post_text_and_the_model_reasons_are_shown(db, monkeypatch):
    at = start(db, monkeypatch)
    rev = at.tabs[1]
    assert "Interview prep\nbody text of post 3" in [t.value for t in rev.get("text")]
    captions = [c.value for c in rev.caption]
    for reason in ("topic reason 3", "difficulty reason 3", "type reason 3", "time reason 3", "alignment reason 3"):
        assert any(reason in c for c in captions)
    assert any("Clustering put this post in: Chunking" in c for c in captions)
    md = [m.value for m in rev.markdown]
    assert any("**Confidence:** 0.61" in m and "below the cut-off" in m and "topic differs" in m for m in md)
    assert any("**AI-engineer score:** 3.0/10 (Low relevance)" in m for m in md)
    assert any("basis: `post_only`" in m for m in md)


def test_the_dropdowns_start_at_the_models_labels(db, monkeypatch):
    at = start(db, monkeypatch)
    assert at.selectbox(key="rv_sub_3").value == "careers.interviews"
    assert at.selectbox(key="rv_diff_3").value == "intermediate"
    assert at.selectbox(key="rv_type_3").value == "interview_question_set"
    assert "Career Development > Interviews" in at.selectbox(key="rv_sub_3").options


def test_the_review_tab_shows_both_links_when_the_post_has_one(db, monkeypatch):
    at = start(db, monkeypatch)
    assert links(at.tabs[1]) == [("Open post", F.url_of(3)), ("Open resource", "https://lnkd.in/c1")]


def test_a_post_without_a_resource_link_shows_only_the_post_link(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_skip_3").click().run()
    at.button(key="rv_skip_6").click().run()
    assert review_title(at) == "Cheap tricks"
    assert links(at.tabs[1]) == [("Open post", F.url_of(4))]


def test_several_links_are_listed_in_an_expander(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_skip_3").click().run()
    assert review_title(at) == "Tips & $tricks* 100%"
    assert [e.label for e in at.tabs[1].get("expander")][0] == "All 2 links found in the post"


# ============================================================ Review: buttons

def test_approve_saves_an_empty_correction_and_moves_on(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_ok_3").click().run()
    saved = reviews(db)
    assert len(saved) == 1 and saved[0]["item_id"] == "3" and saved[0]["corrected_fields"] == "{}"
    assert saved[0]["use_as_example"] == 1
    assert review_title(at) == "Tips & $tricks* 100%"
    assert at.tabs[1].success[0].value == "Approved."
    assert review_counts(at) == "4 posts flagged by the model · 1 reviewed · 3 left in this queue"


def test_the_note_and_the_example_box_are_saved(db, monkeypatch):
    at = start(db, monkeypatch)
    at.text_input(key="rv_note_3").set_value("clearly an interview post")
    at.checkbox(key="rv_ex_3").uncheck()
    at.button(key="rv_ok_3").click().run()
    saved = reviews(db)[0]
    assert saved["note"] == "clearly an interview post" and saved["use_as_example"] == 0


def test_save_fix_stores_only_the_changed_labels(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_fix_3").click().run()
    saved = reviews(db)
    assert len(saved) == 1 and json.loads(saved[0]["corrected_fields"]) == {"difficulty": "advanced"}
    assert at.tabs[1].success[0].value == "Saved: {'difficulty': 'advanced'}"
    assert review_title(at) == "Tips & $tricks* 100%"


def test_a_fix_shows_up_in_the_library(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_sub_3").select("rag.chunking").run()
    at.selectbox(key="rv_type_3").select("tutorial").run()
    at.button(key="rv_fix_3").click().run()
    row = table(at)[table(at)["Title"] == "Interview prep"].iloc[0]
    assert (row["Topic"], row["Subtopic"], row["Type"], row["Status"]) == (
        "Retrieval-Augmented Generation", "Chunking", "tutorial", "reviewed")


def test_save_fix_with_nothing_changed_warns_and_saves_nothing(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_fix_3").click().run()
    assert "Nothing differs from the model's labels" in at.tabs[1].warning[0].value
    assert reviews(db) == [] and review_title(at) == "Interview prep"


def test_approve_after_changing_a_label_warns_and_saves_nothing(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_ok_3").click().run()
    assert "Click Save fix" in at.tabs[1].warning[0].value
    assert reviews(db) == [] and review_title(at) == "Interview prep"


def test_skip_saves_nothing_and_shows_the_next_post(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_skip_3").click().run()
    assert reviews(db) == [] and review_title(at) == "Tips & $tricks* 100%"
    assert review_counts(at) == "4 posts flagged by the model · 0 reviewed · 3 left in this queue · 1 skipped"


def test_skipped_posts_can_be_brought_back(db, monkeypatch):
    at = start(db, monkeypatch)
    for aid in QUEUE:
        at.button(key=f"rv_skip_{aid}").click().run()
    assert "Only skipped posts are left" in at.tabs[1].info[0].value
    at.button(key="rv_unskip").click().run()
    assert review_title(at) == "Interview prep" and "skipped" not in review_counts(at)


def test_the_queue_empties_after_every_post_is_decided(db, monkeypatch):
    at = start(db, monkeypatch)
    for aid in QUEUE:
        at.button(key=f"rv_ok_{aid}").click().run()
    assert at.tabs[1].info[0].value == "Nothing left to review."
    assert len(reviews(db)) == 4 and review_counts(at).endswith("4 reviewed · 0 left in this queue")


def test_reviewed_posts_can_be_shown_again_and_say_what_was_decided(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_fix_3").click().run()
    at.checkbox(key="rv_include").check().run()
    assert review_title(at) == "Interview prep"
    assert "You reviewed this before: {'difficulty': 'advanced'}" in at.tabs[1].info[0].value
    assert at.selectbox(key="rv_diff_3").value == "advanced"                  # the dropdown starts at your correction


def test_a_second_review_of_the_same_post_wins(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_fix_3").click().run()
    at.checkbox(key="rv_include").check().run()
    at.selectbox(key="rv_diff_3").select("beginner").run()
    at.button(key="rv_fix_3").click().run()
    assert [json.loads(r["corrected_fields"]) for r in reviews(db)] == [{"difficulty": "advanced"}, {"difficulty": "beginner"}]
    at.selectbox(key="lib_sort").select("Title (A to Z)").run()
    row = table(at)[table(at)["Title"] == "Interview prep"].iloc[0]
    assert row["Difficulty"] == "beginner"


def test_the_model_labels_are_never_overwritten(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_fix_3").click().run()
    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT difficulty FROM classifications WHERE item_id = '3'").fetchone()[0] == "intermediate"
    finally:
        con.close()


# ============================================================ more screens

def test_clicking_a_table_row_shows_its_summary_and_links(db, monkeypatch):
    at = start(db, monkeypatch)
    at.session_state["lib_table"] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    at.run()
    assert not at.exception
    text = [m.value for m in at.tabs[0].markdown]
    assert r"**Tips \& \$tricks\* 100%**" in text and "Summary of post 6." in text
    assert links(at.tabs[0]) == [("Open post", F.url_of(6)), ("Open resource", "https://lnkd.in/f1")]
    assert "Click a row" not in " ".join(c.value for c in at.tabs[0].caption)
    assert "All 2 links found in the post" in [e.label for e in at.tabs[0].get("expander")]


def test_clicking_a_row_of_a_post_without_resource_links_shows_only_the_post_link(db, monkeypatch):
    at = start(db, monkeypatch)
    at.session_state["lib_table"] = {"selection": {"rows": [2], "columns": [], "cells": []}}
    at.run()
    assert links(at.tabs[0]) == [("Open post", F.url_of(2))]


def test_the_default_filters_show_everything(db, monkeypatch):
    at = start(db, monkeypatch)
    assert at.slider(key="lib_score").value == (0.0, 10.0) and at.slider(key="lib_conf").value == 0.0
    assert at.selectbox(key="lib_status").value == "all" and at.radio(key="lib_view").value == "Table"
    assert count_text(at) == "**6 of 6 posts**"


def test_the_dropdowns_start_at_the_labels_of_the_post_shown(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_skip_3").click().run()
    assert at.selectbox(key="rv_sub_6").value == "rag.evaluation"
    assert at.selectbox(key="rv_diff_6").value == "intermediate"
    assert at.selectbox(key="rv_type_6").value == "tutorial"


def test_reviewed_posts_start_at_all_the_corrected_labels(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_sub_3").select("rag.chunking").run()
    at.selectbox(key="rv_diff_3").select("expert").run()
    at.selectbox(key="rv_type_3").select("news").run()
    at.button(key="rv_fix_3").click().run()
    at.checkbox(key="rv_include").check().run()
    assert (at.selectbox(key="rv_sub_3").value, at.selectbox(key="rv_diff_3").value,
            at.selectbox(key="rv_type_3").value) == ("rag.chunking", "expert", "news")


def test_approving_a_post_that_already_has_a_fix_warns_instead_of_erasing_the_fix(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.button(key="rv_fix_3").click().run()
    at.checkbox(key="rv_include").check().run()
    at.button(key="rv_ok_3").click().run()                       # the dropdown still shows the correction
    assert "Click Save fix" in at.tabs[1].warning[0].value
    assert len(reviews(db)) == 1


def test_the_review_tab_shows_the_key_points_and_the_summary(db, monkeypatch):
    at = start(db, monkeypatch)
    md = [m.value for m in at.tabs[1].markdown]
    assert "- first point\n- second point" in md
    assert "Summary of post 3." in md


def test_the_note_and_the_example_box_are_saved_with_a_fix_too(db, monkeypatch):
    at = start(db, monkeypatch)
    at.selectbox(key="rv_diff_3").select("advanced").run()
    at.text_input(key="rv_note_3").set_value("too easy a label")
    at.checkbox(key="rv_ex_3").uncheck()
    at.button(key="rv_fix_3").click().run()
    saved = reviews(db)[0]
    assert saved["note"] == "too easy a label" and saved["use_as_example"] == 0
    assert json.loads(saved["corrected_fields"]) == {"difficulty": "advanced"}


def test_the_links_expander_lists_every_link_of_that_post(db, monkeypatch):
    at = to_cards(start(db, monkeypatch))
    boxes = at.tabs[0].get("expander")
    shown = [" ".join(m.value for m in b.markdown) for b in boxes]
    assert len(shown) == 2
    assert all(x in shown[0] for x in ("https://lnkd.in/a1", "https://example.com/two", "https://lnkd.in/a3"))
    assert all(x in shown[1] for x in ("https://lnkd.in/f1", "https://example.com/f%282%29"))     # brackets are encoded


# ============================================================ Step 17: the download buttons

def downloads(block):
    return [(b.proto.label, b.proto.disabled) for b in block.get("download_button")]


def test_the_library_has_csv_and_json_download_buttons(db, monkeypatch):
    at = start(db, monkeypatch)
    assert downloads(at.tabs[0]) == [("⬇ These 6 posts as CSV", False), ("⬇ These 6 posts as JSON", False)]
    assert [b.key for b in at.tabs[0].get("download_button")] == ["lib_dl_csv", "lib_dl_json"]


def test_the_library_downloads_follow_the_filters(db, monkeypatch):
    at = start(db, monkeypatch)
    at.multiselect(key="lib_topics").select("careers").run()
    assert downloads(at.tabs[0]) == [("⬇ These 1 posts as CSV", False), ("⬇ These 1 posts as JSON", False)]


def test_nothing_to_download_when_the_filters_match_nothing(db, monkeypatch):
    at = start(db, monkeypatch)
    at.text_input(key="lib_text").set_value("zzzznothing").run()
    assert downloads(at.tabs[0]) == [("⬇ These 0 posts as CSV", True), ("⬇ These 0 posts as JSON", True)]


def test_the_review_tab_downloads_are_off_until_you_review_something(db, monkeypatch):
    at = start(db, monkeypatch)
    assert downloads(at.tabs[1]) == [("⬇ My 0 reviews as CSV", True), ("⬇ My 0 reviews as JSON", True)]
    assert [b.key for b in at.tabs[1].get("download_button")] == ["rv_dl_csv", "rv_dl_json"]


def test_the_review_downloads_count_every_review_whatever_the_library_filters_say(db, monkeypatch):
    at = start(db, monkeypatch)
    at.button(key="rv_ok_3").click().run()
    at.button(key="rv_ok_6").click().run()
    assert downloads(at.tabs[1]) == [("⬇ My 2 reviews as CSV", False), ("⬇ My 2 reviews as JSON", False)]
    at.multiselect(key="lib_topics").select("careers").run()
    assert downloads(at.tabs[1]) == [("⬇ My 2 reviews as CSV", False), ("⬇ My 2 reviews as JSON", False)]
    assert downloads(at.tabs[0])[0] == ("⬇ These 1 posts as CSV", False)
