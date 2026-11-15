"""Step 16: the dashboard (Library and Review tabs).

In simple terms: a small web page on your own computer. It reads the database made by
step15_load_sqlite.py and shows your saved posts. Nothing leaves the machine and no model is
called. This file only draws the screens; every decision is made by dashboard_data.py.

Start it (from the repo root, after `pip install streamlit`):

    streamlit run pipeline/step16_dashboard.py
    streamlit run pipeline/step16_dashboard.py -- --db data/library.sqlite     (another database)

then open http://localhost:8501 (it opens by itself). The database path can also come from the
environment variable LISC_DB.

  Library  filter by topic, difficulty, type, AI-engineer score and band, time, confidence and
           review status; search the title and summary; see a table or cards. Every post has
           "Open post" and (when the post contains links) "Open resource" links. Links are only
           opened by you, in your browser: the dashboard never fetches them.
  Review   the posts the model was unsure about, lowest confidence first. Read the post next to
           the model's labels and its reasons, then Approve (the model was right), Save fix
           (you changed the dropdowns) or Skip (decide later; nothing is saved). Fixes go to the
           `reviews` table; the model's own labels are never overwritten.

Differences from the design (LLD 14): no semantic search box (no vector store yet), no "Mark
done" (needs a progress table), and the Planner, Insights and Runs tabs come in chat 4.
"""
from __future__ import annotations

import os
import sys
from contextlib import closing

import streamlit as st

import dashboard_data as D
import step17_export as X

PAGE_SIZE = 20
REASON_TEXT = {
    "low_confidence": "the confidence is below the cut-off",
    "other_general": "the model found no fitting subtopic (other.general)",
    "topic_disagrees_with_clustering": "the topic differs from the topic clustering gave this post",
}
STATUS_BADGE = {"needs_review": "⚠ needs review", "reviewed": "✎ reviewed", "auto_accepted": "✓ auto-accepted"}


# ------------------------------------------------------------------ small pieces

def link_buttons(row: dict, key: str) -> None:
    """'Open post' always; 'Open resource' only when the post contains a link."""
    cols = st.columns(2)
    with cols[0]:
        st.link_button("Open post", row["post_url"], key=f"post_{key}")
    with cols[1]:
        if row["resource_link"]:
            st.link_button("Open resource", row["resource_link"], key=f"res_{key}")


def other_links(row: dict) -> None:
    links = row["resource_links_list"]
    if len(links) > 1:
        with st.expander(f"All {len(links)} links found in the post"):
            st.markdown(D.md_link_list(links))


def facts_line(r: dict) -> str:
    return (f"`{r['difficulty']}` · `{r['content_type']}` · cover {D.format_minutes(r['cover_minutes'])} · "
            f"understand {D.format_minutes(r['understand_minutes'])} · AI-engineer score {r['alignment_score']}/10 · "
            f"confidence {r['confidence']}")


# ------------------------------------------------------------------ Library

def library_filters(rows: list[dict], opts: dict) -> D.Filters:
    topic_names = dict(opts["topics"])
    topics = st.multiselect("Topic", [t for t, _ in opts["topics"]], format_func=lambda t: topic_names[t], key="lib_topics")
    subs = [(s, n) for s, n, t in opts["subtopics"] if not topics or t in topics]
    sub_names = dict(subs)
    subtopics = st.multiselect("Subtopic", list(sub_names), format_func=lambda s: sub_names[s], key="lib_subtopics")
    difficulties = st.multiselect("Difficulty", opts["difficulties"], key="lib_difficulties")
    content_types = st.multiselect("Type", opts["content_types"], key="lib_types")
    min_score, max_score = st.slider("AI-engineer alignment score", 0.0, 10.0, (0.0, 10.0), step=0.5, key="lib_score")
    bands = st.multiselect("Alignment band", opts["bands"], key="lib_bands")
    cover = st.multiselect("Cover time", opts["cover_buckets"], key="lib_cover")
    understand = st.multiselect("Understand time", opts["understand_buckets"], key="lib_understand")
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, step=0.05, key="lib_conf")
    status = st.selectbox("Review status", list(D.STATUS_CHOICES), format_func=D.STATUS_LABELS.get, key="lib_status")
    text = st.text_input("Search title, summary, author", key="lib_text")
    return D.Filters(topics=tuple(topics), subtopics=tuple(subtopics), difficulties=tuple(difficulties),
                     content_types=tuple(content_types), bands=tuple(bands), cover_buckets=tuple(cover),
                     understand_buckets=tuple(understand), min_score=min_score, max_score=max_score,
                     min_confidence=min_conf, status=status, text=text)


def show_table(rows: list[dict]) -> None:
    event = st.dataframe(
        D.table_rows(rows), hide_index=True, width="stretch", key="lib_table",
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Post": st.column_config.LinkColumn("Post", display_text="Open post"),
            "Resource": st.column_config.LinkColumn("Resource", display_text="Open resource"),
            "Summary": st.column_config.TextColumn("Summary", width="large"),
            "AI-engineer score": st.column_config.NumberColumn(format="%.1f"),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
        })
    picked = event.selection.rows if event and event.selection else []
    r = D.pick_row(rows, picked)
    if r is not None:
        with st.container(border=True):
            st.markdown(f"**{D.md_escape(r['title'])}**")
            st.caption(f"{r['author'] or ''} · {r['topic_name']} › {r['subtopic_name']}")
            st.markdown(facts_line(r))
            st.markdown(D.md_escape(r["summary"]))
            link_buttons(r, f"sel{r['id']}")
            other_links(r)
    else:
        st.caption("Click a row to see its summary and links below the table.")


def show_cards(rows: list[dict]) -> None:
    pages = D.page_count(len(rows), PAGE_SIZE)
    page = int(st.number_input("Page", 1, pages, 1, key="lib_page")) if pages > 1 else 1
    for r in D.page_of(rows, page, PAGE_SIZE):
        with st.container(border=True):
            st.markdown(f"**{D.md_escape(r['title'])}**")
            st.caption(f"{r['author'] or ''} · {r['topic_name']} › {r['subtopic_name']} · {STATUS_BADGE[r['status']]}")
            st.markdown(facts_line(r))
            st.markdown(D.md_escape(r["summary"]))
            link_buttons(r, f"card{r['id']}")
            other_links(r)
    if pages > 1:
        st.caption(f"Page {page} of {pages}")


def show_downloads(specs: list[dict]) -> None:
    """Draw the download buttons that step17_export.py describes (same files as the script writes)."""
    for column, spec in zip(st.columns([1, 1, 2]), specs):
        column.download_button(spec["label"], spec["data"], file_name=spec["file_name"], mime=spec["mime"],
                               disabled=spec["disabled"], key=spec["key"])


def library_tab(rows: list[dict]) -> None:
    opts = D.filter_options(rows)
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Filters")
        f = library_filters(rows, opts)
    with right:
        shown = D.apply_filters(rows, f)
        top = st.columns([2, 2, 3])
        with top[0]:
            view = st.radio("View", ["Table", "Cards"], horizontal=True, key="lib_view")
        with top[1]:
            sort = st.selectbox("Sort by", list(D.SORTS), key="lib_sort")
        with top[2]:
            st.markdown(f"**{len(shown)} of {len(rows)} posts**")
        shown = D.sort_rows(shown, sort)
        show_downloads(X.library_downloads(shown))
        if not shown:
            st.info("No posts match these filters.")
        elif view == "Table":
            show_table(shown)
        else:
            show_cards(shown)


# ------------------------------------------------------------------ Review

def review_tab(con, rows: list[dict]) -> None:
    c = D.counts(rows)
    show_downloads(X.correction_downloads(rows))
    include = st.checkbox("Also show posts I already reviewed", key="rv_include")
    skipped = st.session_state.setdefault("rv_skipped", [])
    queue = [r for r in D.review_queue(rows, include_reviewed=include) if r["id"] not in skipped]
    st.caption(f"{c['flagged_by_model']} posts flagged by the model · {c['reviewed']} reviewed · "
               f"{len(queue)} left in this queue" + (f" · {len(skipped)} skipped" if skipped else ""))
    flash = st.session_state.pop("rv_flash", None)
    if flash:
        st.success(flash)
    if not queue:
        st.info("Nothing left to review." if not skipped else "Only skipped posts are left.")
        if skipped and st.button("Bring back the skipped posts", key="rv_unskip"):
            st.session_state["rv_skipped"] = []
            st.rerun()
        return
    r = queue[0]
    d = D.review_detail(con, r["classification_id"])
    key = r["id"]

    st.subheader(r["title"] or "(no title)")
    st.caption(f"{r['author'] or ''} · saved {r['saved_at'] or 'date unknown'} · model {d['model']}")
    link_buttons(r, f"rv{key}")
    other_links(r)
    if r["reviewed"]:
        st.info(f"You reviewed this before: {r['corrected_fields'] or 'approved as it was'}")

    left, right = st.columns(2)
    with left:
        st.markdown("**The post**")
        with st.container(height=520, border=True):
            st.text(d["body_text"])
    with right:
        st.markdown("**Labels** (change a dropdown to correct it)")
        subs = D.subtopic_choices(con, d["taxonomy_version"])
        sub_ids = [s for s, _ in subs]
        sub_label = dict(subs)
        chosen_sub = st.selectbox("Subtopic (the topic follows from it)", sub_ids, index=sub_ids.index(r["subtopic_id"]),
                                  format_func=lambda s: sub_label[s], key=f"rv_sub_{key}")
        st.caption(f"Model's reason: {d['rationales'].get('topic', '')}")
        if d["clustering_subtopic_id"]:
            st.caption(f"Clustering put this post in: {d['clustering_name'] or d['clustering_subtopic_id']}")
        chosen_diff = st.selectbox("Difficulty", D.DIFFICULTY_IDS, index=D.DIFFICULTY_IDS.index(r["difficulty"]),
                                   key=f"rv_diff_{key}")
        st.caption(f"Model's reason: {d['rationales'].get('difficulty', '')}")
        chosen_type = st.selectbox("Type", D.CONTENT_TYPE_IDS, index=D.CONTENT_TYPE_IDS.index(r["content_type"]),
                                   key=f"rv_type_{key}")
        st.caption(f"Model's reason: {d['rationales'].get('content_type', '')}")

        t = d["time"]
        st.markdown(f"**Time:** cover {D.format_minutes(r['cover_minutes'])}, understand "
                    f"{D.format_minutes(r['understand_minutes'])} · basis: `{t.get('basis', '?')}`"
                    + (f" · the post states: {t['stated_size']}" if t.get("stated_size") else ""))
        st.caption(f"Model's reason: {d['rationales'].get('time', '')}")
        st.markdown(f"**AI-engineer score:** {r['alignment_score']}/10 ({r['alignment_band']})")
        st.caption(f"Model's reason: {d['rationales'].get('alignment', '')}")
        why = "; ".join(REASON_TEXT.get(x, x) for x in d["review_reasons"]) or "none"
        st.markdown(f"**Confidence:** {r['confidence']} · flagged because: {why}")
        st.markdown("**Summary**")
        st.write(r["summary"])
        if d["key_points"]:
            st.markdown("\n".join(f"- {D.md_escape(k)}" for k in d["key_points"]))
        with st.expander("Technical details"):
            st.write({"warnings": d["warnings"], "retry reasons": d["retry_reasons"], "attempts": d["attempts"],
                      "alignment sub-scores": d["alignment_subscores"], "confidence parts": d["confidence_parts"],
                      "confidence the run wrote": d["confidence_as_run"]})

    note = st.text_input("Note (optional)", key=f"rv_note_{key}")
    example = st.checkbox("Use this as an example later", value=True, key=f"rv_ex_{key}")
    changed = {"subtopic_id": chosen_sub, "difficulty": chosen_diff, "content_type": chosen_type}
    differs = any(changed[k] != r[f"model_{k}"] for k in D.EDITABLE)
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve (the model was right)", type="primary", key=f"rv_ok_{key}"):
        if differs:
            st.warning("You changed a label. Click Save fix to keep your change, or set the dropdowns back to approve.")
        else:
            D.approve_review(con, r["id"], note, example)
            st.session_state["rv_flash"] = "Approved."
            st.rerun()
    if b2.button("💾 Save fix", key=f"rv_fix_{key}"):
        try:
            _, saved = D.save_fix(con, r["id"], changed, note, example)
        except D.DashboardError as exc:
            st.warning(str(exc))
        else:
            st.session_state["rv_flash"] = f"Saved: {saved}"
            st.rerun()
    if b3.button("⏭ Skip", key=f"rv_skip_{key}"):
        skipped.append(r["id"])
        st.rerun()


# ------------------------------------------------------------------ the page

def main(argv: list[str] | None = None, environ=None) -> None:
    st.set_page_config(page_title="LinkedIn saved posts", page_icon="📚", layout="wide")
    db_path = D.resolve_db_path(sys.argv[1:] if argv is None else argv, os.environ if environ is None else environ)
    try:
        con = D.connect(db_path)
    except D.DashboardError as exc:
        st.error(str(exc))
        st.stop()
    with closing(con):
        rows = D.load_library(con)
        st.title("LinkedIn saved posts")
        st.caption(f"{len(rows)} classified posts · database {db_path}")
        if not rows:
            st.warning("The database has no classified posts yet. Load a results file with step15_load_sqlite.py.")
            st.stop()
        lib, rev = st.tabs(["📚 Library", "🟡 Review"])
        with lib:
            library_tab(rows)
        with rev:
            review_tab(con, rows)


main()
