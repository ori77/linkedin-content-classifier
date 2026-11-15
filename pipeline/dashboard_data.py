"""The data side of the dashboard (Step 16): everything except the screens.

In simple terms: the dashboard (step16_dashboard.py) only draws things. All the thinking is
here, in plain functions that can be tested without opening a browser:

  * connect              open the database made by step15_load_sqlite.py
  * load_library         every classified post as one dict, with YOUR corrections laid over
                         the model's labels (the library shows your label when you fixed one)
  * Filters / apply_filters / sort_rows / filter_options     what the Library tab filters on
  * review_queue         the posts flagged for review, lowest confidence first
  * review_detail        everything the Review tab shows next to the labels (text, reasons, ...)
  * approve_review / save_fix     save your decision into the `reviews` table
  * table_rows / md_escape / format_minutes    small helpers for showing things

How a correction works: the model's own labels stay untouched in `classifications`. A review
stores only the fields you changed (`corrected_fields`, for example {"difficulty": "advanced"}).
An approval stores an empty set ({}), meaning "the model was right". When a post has several
reviews, the newest one counts. Only three labels can be corrected here: subtopic (the topic
follows from it), difficulty and content type. The time, the alignment score and the summary
cannot be edited (not in the plan for chat 3).

Links: every row carries `post_url` and `resource_link` (the first link found in the post, or
None). Links are only shown, never fetched.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rubrics import CONTENT_TYPE_IDS, DIFFICULTY_IDS
from step15_load_sqlite import SCHEMA_VERSION

DEFAULT_DB = "data/library.sqlite"
EDITABLE = ("subtopic_id", "difficulty", "content_type")        # the labels a review may correct
OTHER_ID = "other.general"
STATUS_CHOICES = ("all", "needs_review", "reviewed", "auto_accepted")
STATUS_LABELS = {"all": "All posts", "needs_review": "Needs review (not reviewed yet)",
                 "reviewed": "Reviewed by me", "auto_accepted": "Accepted automatically"}
SORTS = {                                  # label -> (field, biggest first?)
    "Alignment score (high first)": ("alignment_score", True),
    "Confidence (low first)": ("confidence", False),
    "Cover time (short first)": ("cover_minutes", False),
    "Understand time (short first)": ("understand_minutes", False),
    "Saved date (newest first)": ("saved_at", True),
    "Title (A to Z)": ("title_sort", False),
}
DEFAULT_SORT = "Alignment score (high first)"


class DashboardError(Exception):
    """Something the dashboard can explain to the user in one sentence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_db_path(argv: list[str], environ: Mapping[str, str]) -> str:
    """The database to open: `--db path` on the command line, else $LISC_DB, else the default.

    Streamlit passes everything after `--` to the script:  streamlit run <script> -- --db path
    """
    for i, arg in enumerate(argv):
        if arg == "--db" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--db="):
            return arg[len("--db="):]
    return environ.get("LISC_DB") or DEFAULT_DB


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database made by step15_load_sqlite.py, or say clearly what is wrong."""
    path = Path(path)
    if not path.is_file():
        raise DashboardError(
            f"No database at {path}. Build it first with: python pipeline/step15_load_sqlite.py "
            f"--jsonl <results file> --csv input-data/posts_with_save_dates.csv "
            f"--taxonomy pipeline/taxonomy/taxonomy.v1.yaml --db {path}")
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        version = con.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.Error as exc:
        raise DashboardError(f"{path} is not a usable database: {exc}") from exc
    if version != SCHEMA_VERSION:
        con.close()
        raise DashboardError(f"{path} was not made by step15_load_sqlite.py "
                             f"(schema version {version}, expected {SCHEMA_VERSION})")
    return con


# ------------------------------------------------------------------ reading the library

def _json(text: str | None, default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except (TypeError, ValueError):
        return default


def format_minutes(minutes: float | int | None) -> str:
    """90 -> '1h 30m', 45 -> '45m', 1500 -> '1d 1h' (a day counts as 24 hours here)."""
    if minutes is None:
        return "?"
    m = int(round(minutes))
    if m < 60:
        return f"{m}m"
    if m < 24 * 60:
        h, rest = divmod(m, 60)
        return f"{h}h {rest}m" if rest else f"{h}h"
    d, rest = divmod(m, 24 * 60)
    h = rest // 60
    return f"{d}d {h}h" if h else f"{d}d"


def _names(con: sqlite3.Connection) -> dict[tuple[str, str], tuple[str, str, str]]:
    """(taxonomy_version, subtopic_id) -> (topic_id, topic_name, subtopic_name)."""
    out = {}
    for r in con.execute("""SELECT s.taxonomy_version v, s.id sid, s.topic_id tid, t.name tname, s.name sname
                            FROM subtopics s JOIN topics t
                              ON t.taxonomy_version = s.taxonomy_version AND t.id = s.topic_id"""):
        out[(r["v"], r["sid"])] = (r["tid"], r["tname"], r["sname"])
    return out


def _latest_reviews(con: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    out = {}
    for r in con.execute("SELECT * FROM reviews ORDER BY created_at, rowid"):
        out[r["item_id"]] = r                    # the newest review of an item wins
    return out


def load_library(con: sqlite3.Connection) -> list[dict]:
    """Every classified post, newest corrections applied. One dict per post.

    Besides the columns of the `library` view the dict has:
      model_subtopic_id, model_difficulty, model_content_type   what the model said
      subtopic_id, topic_id, difficulty, content_type            what counts now (your fix wins)
      reviewed, corrected_fields, review_note                    your review, if any
      review_id, reviewed_at, use_as_example                     details of that review (None without one)
      status                  needs_review | reviewed | auto_accepted
      resource_links_list     all resource links as a list
    """
    names = _names(con)
    reviews = _latest_reviews(con)
    rows = []
    for r in con.execute("""SELECT l.*, c.taxonomy_version, c.needs_review AS model_flagged
                            FROM library l JOIN classifications c ON c.id = l.classification_id"""):
        row = dict(r)
        row["resource_links_list"] = _json(row["resource_links"], [])
        row["model_subtopic_id"] = row["subtopic_id"]
        row["model_difficulty"] = row["difficulty"]
        row["model_content_type"] = row["content_type"]
        row["model_flagged"] = bool(row["model_flagged"])
        row["review_reasons"] = _json(row["review_reasons"], [])
        review = reviews.get(row["id"])
        row["reviewed"] = review is not None
        row["corrected_fields"] = {}
        row["review_note"] = None
        row["review_id"] = row["reviewed_at"] = row["use_as_example"] = None
        if review is not None:
            fixes = _json(review["corrected_fields"], {})
            row["review_note"] = review["note"]
            row["review_id"] = review["id"]
            row["reviewed_at"] = review["created_at"]
            row["use_as_example"] = bool(review["use_as_example"])
            for key, value in (fixes.items() if isinstance(fixes, dict) else []):
                if key == "subtopic_id" and (row["taxonomy_version"], value) in names:
                    tid, tname, sname = names[(row["taxonomy_version"], value)]
                    row.update(subtopic_id=value, topic_id=tid, topic_name=tname, subtopic_name=sname)
                elif key == "difficulty" and value in DIFFICULTY_IDS:
                    row["difficulty"] = value
                elif key == "content_type" and value in CONTENT_TYPE_IDS:
                    row["content_type"] = value
                else:
                    continue                     # an unknown field or value is ignored, not shown
                row["corrected_fields"][key] = value
        row["status"] = ("reviewed" if row["reviewed"]
                         else "needs_review" if row["model_flagged"] else "auto_accepted")
        row["title_sort"] = (row["title"] or "").lower()
        rows.append(row)
    rows.sort(key=lambda x: x["activity_id"])
    return rows


# ------------------------------------------------------------------ filters, sorting, options

@dataclass(frozen=True)
class Filters:
    """What the Library tab filters on. An empty choice means "no filter on this"."""
    topics: tuple[str, ...] = ()
    subtopics: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    bands: tuple[str, ...] = ()
    cover_buckets: tuple[str, ...] = ()
    understand_buckets: tuple[str, ...] = ()
    min_score: float = 0.0
    max_score: float = 10.0
    min_confidence: float = 0.0
    status: str = "all"
    text: str = ""


def apply_filters(rows: list[dict], f: Filters) -> list[dict]:
    if f.status not in STATUS_CHOICES:
        raise DashboardError(f"unknown status filter '{f.status}'")
    words = f.text.lower().split()
    out = []
    for r in rows:
        if f.topics and r["topic_id"] not in f.topics:
            continue
        if f.subtopics and r["subtopic_id"] not in f.subtopics:
            continue
        if f.difficulties and r["difficulty"] not in f.difficulties:
            continue
        if f.content_types and r["content_type"] not in f.content_types:
            continue
        if f.bands and r["alignment_band"] not in f.bands:
            continue
        if f.cover_buckets and r["cover_bucket"] not in f.cover_buckets:
            continue
        if f.understand_buckets and r["understand_bucket"] not in f.understand_buckets:
            continue
        if not f.min_score <= r["alignment_score"] <= f.max_score:
            continue
        if r["confidence"] < f.min_confidence:
            continue
        if f.status != "all" and r["status"] != f.status:
            continue
        if words:
            haystack = " ".join(str(r.get(k) or "") for k in
                                ("title", "summary", "author", "topic_name", "subtopic_name")).lower()
            if not all(w in haystack for w in words):
                continue
        out.append(r)
    return out


def sort_rows(rows: list[dict], sort: str = DEFAULT_SORT) -> list[dict]:
    """Sort by one of SORTS. Posts with no value for the field (for example no save date) go last."""
    if sort not in SORTS:
        raise DashboardError(f"unknown sort '{sort}'")
    field, biggest_first = SORTS[sort]
    have = sorted((r for r in rows if r.get(field) is not None), key=lambda r: r["activity_id"])
    have.sort(key=lambda r: r[field], reverse=biggest_first)
    return have + [r for r in rows if r.get(field) is None]


def filter_options(rows: list[dict]) -> dict[str, list]:
    """What the filter widgets offer, from the posts that exist (in a sensible order)."""
    def ordered(field: str, by: str, reverse: bool = False) -> list[str]:
        best: dict[str, float] = {}
        for r in rows:
            v = r[field]
            best[v] = max(best.get(v, float("-inf")), r[by]) if reverse else min(best.get(v, float("inf")), r[by])
        return sorted(best, key=lambda k: (best[k], k), reverse=reverse)

    topics = {r["topic_id"]: r["topic_name"] for r in rows}
    subs = {r["subtopic_id"]: (r["subtopic_name"], r["topic_id"]) for r in rows}
    return {
        "topics": sorted(((t, n) for t, n in topics.items()), key=lambda x: (x[1].lower(), x[0])),
        "subtopics": sorted(((s, n, t) for s, (n, t) in subs.items()), key=lambda x: (x[1].lower(), x[0])),
        "difficulties": [d for d in DIFFICULTY_IDS if any(r["difficulty"] == d for r in rows)],
        "content_types": [c for c in CONTENT_TYPE_IDS if any(r["content_type"] == c for r in rows)],
        "bands": ordered("alignment_band", "alignment_score", reverse=True),
        "cover_buckets": ordered("cover_bucket", "cover_minutes"),
        "understand_buckets": ordered("understand_bucket", "understand_minutes"),
    }


def pick_row(rows: list[dict], picked: list[int]) -> dict | None:
    """The row the user clicked in the table (`picked` = row numbers from Streamlit), or None."""
    return rows[picked[0]] if picked and 0 <= picked[0] < len(rows) else None


def page_count(n_rows: int, size: int) -> int:
    """How many pages of `size` cards are needed for n_rows (at least 1, so an empty list still has a page)."""
    return max(1, -(-n_rows // size))


def page_of(rows: list[dict], page: int, size: int) -> list[dict]:
    """The rows of page number `page` (counted from 1; a page outside the range is moved to the nearest one)."""
    page = min(max(1, page), page_count(len(rows), size))
    return rows[(page - 1) * size: page * size]


def counts(rows: list[dict]) -> dict[str, int]:
    return {"total": len(rows),
            "flagged_by_model": sum(r["model_flagged"] for r in rows),
            "reviewed": sum(r["reviewed"] for r in rows),
            "to_review": sum(r["status"] == "needs_review" for r in rows)}


# ------------------------------------------------------------------ showing things

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$&:])")


_LINE_START = re.compile(r"^(\s*)(\d+)\.|^(\s*)-", re.MULTILINE)


def md_escape(text: str | None) -> str:
    """Make post text safe to show through st.markdown (no accidental bold, links, LaTeX, HTML or lists)."""
    text = _MARKDOWN_SPECIAL.sub(r"\\\1", (text or "").replace("\r", ""))
    return _LINE_START.sub(lambda m: f"{m.group(1)}{m.group(2)}\\." if m.group(2) else f"{m.group(3)}\\-", text)


def md_link_list(urls: list[str]) -> str:
    """A markdown bullet list of clickable links (brackets and spaces in an address are written as %28 %29 %20)."""
    return "\n".join(
        f"- [{md_escape(u)}]({u.replace('(', '%28').replace(')', '%29').replace(' ', '%20')})" for u in urls)


def table_rows(rows: list[dict]) -> list[dict]:
    """The columns of the table view, in order. Links are plain web addresses (the screen makes them clickable)."""
    return [{
        "Title": r["title"] or "(no title)", "Author": r["author"] or "",
        "Topic": r["topic_name"], "Subtopic": r["subtopic_name"],
        "Difficulty": r["difficulty"], "Type": r["content_type"],
        "Cover": format_minutes(r["cover_minutes"]), "Understand": format_minutes(r["understand_minutes"]),
        "AI-engineer score": r["alignment_score"], "Band": r["alignment_band"],
        "Confidence": r["confidence"],
        "Status": {"needs_review": "needs review", "reviewed": "reviewed",
                   "auto_accepted": "auto-accepted"}[r["status"]],
        "Summary": r["summary"],
        "Post": r["post_url"], "Resource": r["resource_link"],
    } for r in rows]


# ------------------------------------------------------------------ the review queue

def review_queue(rows: list[dict], include_reviewed: bool = False) -> list[dict]:
    """Posts the model flagged, lowest confidence first (ties by post id). Reviewed ones are hidden by default."""
    queue = [r for r in rows if r["model_flagged"] and (include_reviewed or not r["reviewed"])]
    return sorted(queue, key=lambda r: (r["confidence"], r["activity_id"]))


def subtopic_choices(con: sqlite3.Connection, taxonomy_version: str) -> list[tuple[str, str]]:
    """(subtopic id, 'Topic > Subtopic') for the dropdown; other.general is last."""
    out = [(r["sid"], f"{r['tname']} > {r['sname']}") for r in con.execute(
        """SELECT s.id sid, t.name tname, s.name sname FROM subtopics s JOIN topics t
           ON t.taxonomy_version = s.taxonomy_version AND t.id = s.topic_id
           WHERE s.taxonomy_version = ? AND s.id <> ? ORDER BY lower(t.name), lower(s.name), s.id""",
        (taxonomy_version, OTHER_ID))]
    other = con.execute("SELECT name FROM subtopics WHERE taxonomy_version = ? AND id = ?",
                        (taxonomy_version, OTHER_ID)).fetchone()
    if other is not None:
        out.append((OTHER_ID, other["name"]))
    return out


def review_detail(con: sqlite3.Connection, classification_id: str) -> dict:
    """Everything the Review tab shows next to the labels, for one classification."""
    r = con.execute(
        """SELECT c.*, co.body_text, cl.name AS clustering_name
           FROM classifications c
           LEFT JOIN contents co ON co.item_id = c.item_id
           LEFT JOIN subtopics cl ON cl.taxonomy_version = c.taxonomy_version AND cl.id = c.clustering_subtopic_id
           WHERE c.id = ?""", (classification_id,)).fetchone()
    if r is None:
        raise DashboardError(f"no classification '{classification_id}'")
    return {
        "body_text": r["body_text"] or "", "taxonomy_version": r["taxonomy_version"],
        "rationales": _json(r["rationales"], {}), "key_points": _json(r["key_points"], []),
        "warnings": _json(r["warnings"], []), "retry_reasons": _json(r["retry_reasons"], {}),
        "attempts": _json(r["attempts"], {}), "time": _json(r["time_details"], {}),
        "alignment_subscores": _json(r["alignment_subscores"], {}),
        "confidence_parts": _json(r["confidence_parts"], {}),
        "review_reasons": _json(r["review_reasons"], []),
        "clustering_subtopic_id": r["clustering_subtopic_id"], "clustering_name": r["clustering_name"],
        "confidence_as_run": r["confidence_as_run"], "model": r["model"],
    }


# ------------------------------------------------------------------ saving a decision

def _active(con: sqlite3.Connection, item_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM classifications WHERE item_id = ? AND is_active = 1", (item_id,)).fetchone()
    if row is None:
        raise DashboardError(f"post {item_id} has no active classification")
    return row


def _write_review(con: sqlite3.Connection, item_id: str, cls_id: str, corrected: dict,
                  note: str, use_as_example: bool) -> str:
    review_id = "rev_" + uuid.uuid4().hex[:16]
    with con:
        con.execute(
            """INSERT INTO reviews(id, item_id, classification_id, corrected_fields, note, use_as_example, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (review_id, item_id, cls_id, json.dumps(corrected),
             note.strip() or None, int(bool(use_as_example)), utc_now()))
    return review_id


def approve_review(con: sqlite3.Connection, item_id: str, note: str = "", use_as_example: bool = True) -> str:
    """"The model's labels are right." Stores an empty set of corrections. Returns the review id."""
    return _write_review(con, item_id, _active(con, item_id)["id"], {}, note, use_as_example)


def save_fix(con: sqlite3.Connection, item_id: str, chosen: dict[str, str], note: str = "",
             use_as_example: bool = True) -> tuple[str, dict]:
    """Save the labels you chose. Only the ones that differ from the MODEL's labels are stored.

    `chosen` has the keys of EDITABLE. Returns (review id, the stored corrections).
    Raises DashboardError when a value is not allowed or nothing differs from the model.
    """
    cls = _active(con, item_id)
    unknown = [k for k in chosen if k not in EDITABLE]
    if unknown:
        raise DashboardError(f"cannot correct {unknown}; only {list(EDITABLE)}")
    valid_subs = {sid for sid, _ in subtopic_choices(con, cls["taxonomy_version"])}
    allowed = {"subtopic_id": valid_subs, "difficulty": set(DIFFICULTY_IDS), "content_type": set(CONTENT_TYPE_IDS)}
    corrected = {}
    for key, value in chosen.items():
        if value not in allowed[key]:
            raise DashboardError(f"'{value}' is not a valid {key}")
        if value != cls[key]:
            corrected[key] = value
    if not corrected:
        raise DashboardError("Nothing differs from the model's labels. Use Approve if they are right.")
    return _write_review(con, item_id, cls["id"], corrected, note, use_as_example), corrected
