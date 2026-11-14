"""Step 15: load the classified posts into one SQLite database file.

In simple terms: so far every result lived in a .jsonl file and the posts in a CSV. The dashboard
(Step 16) and the exports (Step 17) need one place where everything sits together: the post, the
labels, the confidence, and the links back to LinkedIn. This script builds that place.

What goes where (a simplified version of LLD 5, see the "differences" list at the end):

  items             one row per saved post: author, a short title, dates, the POST LINK
                    (post_url) and the RESOURCE LINKS found in the post (resource_links)
  contents          the post text and its numbers (word count, image, ...)
  classifications   what the model said, one row per run per post. Nothing is overwritten: when a
                    newer run of the same post is loaded, the older row stays and is marked
                    is_active = 0. At most one row per post is active.
  reviews           your corrections (empty now; the Review tab of Step 16 fills it)
  runs              one row per loaded .jsonl file (model, prompt, rubrics, review rule used)
  taxonomy_versions the topic tree, frozen by version
  topics, subtopics the names of the topics and subtopics, so the dashboard can show names
  library (a view)  the table the dashboard reads: active classification + post + both links

Links: the .jsonl has none, so they are joined from the posts CSV by activity_id. They are only
stored and shown. Nothing here (or anywhere) opens a link or reads a web page.

Review rule at load time: the confidence and needs_review of every record are RECOMPUTED from the
numbers the record stores (confidence_parts) with the current review rule file, so a new
threshold needs no new model run. One thing is different from the run itself (open item 54):
when the CLUSTERING could not place a post (it is in other.general), clustering has "no opinion",
so that part is left out of the confidence and the other parts are rescaled. During the run such a
post was scored as if clustering disagreed (0.4), which flagged 42 of 43 of them for no real
reason. The original values stay in the database as confidence_as_run / needs_review_as_run.

Run (from the repo root; the database file is created if missing, and loading again is safe):

  python pipeline/step15_load_sqlite.py --jsonl gold/classifications_full_gemma4-e4b.jsonl ^
      --csv input-data/posts_with_save_dates.csv --taxonomy pipeline/taxonomy/taxonomy.v1.yaml ^
      --db data/library.sqlite

Loading the same file again changes nothing except the recomputed review fields. Loading a newer
run makes its rows the active ones. Your reviews are never touched.

Exit status: 0 = loaded and checked, 1 = loaded but a check failed, 2 = could not load (the
problems are listed together).

Differences from LLD 5 (the prototype is simpler on purpose):
  * the key of an item is the activity_id (as text), not a ulid; post_url plays the role of
    canonical_url; there is no `embeddings` table and no vector tables;
  * contents has no outbound_urls column (the links live on items, in one place);
  * extra: topics and subtopics tables (names), the time basis and the confidence parts, and
    both links in the library view.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import classifier as C
import classify_inputs as I
import review_rule as RV

SCHEMA_VERSION = 1
KIND = "post"
TITLE_CHARS = 90                       # a post has no title; its first line (cut here) stands in for one
UNKNOWN_COLLECTOR = "unknown"
NO_TEXT = "no_text"
CLASSIFIED = "classified"
RESOURCE_SEPARATOR = " | "             # how the posts CSV joins several links in its `urls` column

SCHEMA = """
CREATE TABLE items (
  id                  TEXT PRIMARY KEY,                  -- the activity_id as text
  activity_id         INTEGER NOT NULL UNIQUE,
  post_url            TEXT NOT NULL UNIQUE CHECK (post_url <> ''),   -- the link back to the post
  resource_links      TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(resource_links)),  -- all links found in the post
  resource_link       TEXT,                              -- the first of them (NULL = the post has none)
  resource_link_count INTEGER NOT NULL DEFAULT 0,
  kind                TEXT NOT NULL DEFAULT 'post',
  collector           TEXT NOT NULL,                     -- the export file the post came from
  author              TEXT,
  title               TEXT,
  post_date           DATE,
  saved_at            TIMESTAMP,
  capture_status      TEXT NOT NULL,                     -- captured | saved_after_export
  status              TEXT NOT NULL,                     -- classified | no_text
  duplicate_of        TEXT REFERENCES items(id),
  created_at          TIMESTAMP NOT NULL,
  updated_at          TIMESTAMP NOT NULL
);
CREATE INDEX ix_items_status ON items(status);

CREATE TABLE contents (
  item_id       TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
  body_text     TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  metrics       TEXT NOT NULL CHECK (json_valid(metrics)),
  partial       BOOLEAN NOT NULL DEFAULT 0
);
CREATE INDEX ix_contents_hash ON contents(content_hash);

CREATE TABLE taxonomy_versions (
  version    TEXT PRIMARY KEY,
  sha256     TEXT NOT NULL UNIQUE,
  tree       TEXT NOT NULL CHECK (json_valid(tree)),
  created_at TIMESTAMP NOT NULL
);

CREATE TABLE topics (
  taxonomy_version TEXT NOT NULL REFERENCES taxonomy_versions(version),
  id               TEXT NOT NULL,
  name             TEXT NOT NULL,
  PRIMARY KEY (taxonomy_version, id)
);

CREATE TABLE subtopics (
  taxonomy_version TEXT NOT NULL REFERENCES taxonomy_versions(version),
  id               TEXT NOT NULL,
  topic_id         TEXT NOT NULL,
  name             TEXT NOT NULL,
  PRIMARY KEY (taxonomy_version, id)
);

CREATE TABLE runs (
  id              TEXT PRIMARY KEY,
  command         TEXT,
  config_snapshot TEXT CHECK (config_snapshot IS NULL OR json_valid(config_snapshot)),
  items_ok        INTEGER NOT NULL DEFAULT 0,
  items_failed    INTEGER NOT NULL DEFAULT 0,
  input_tokens    INTEGER NOT NULL DEFAULT 0,
  output_tokens   INTEGER NOT NULL DEFAULT 0,
  est_cost        REAL NOT NULL DEFAULT 0,
  started_at      TIMESTAMP,
  finished_at     TIMESTAMP
);

CREATE TABLE classifications (
  id                    TEXT PRIMARY KEY,
  item_id               TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  topic_id              TEXT NOT NULL,
  subtopic_id           TEXT NOT NULL,
  difficulty            TEXT NOT NULL CHECK (difficulty IN ('beginner','intermediate','advanced','expert')),
  content_type          TEXT NOT NULL,
  cover_minutes         INTEGER NOT NULL CHECK (cover_minutes > 0),
  understand_minutes    INTEGER NOT NULL CHECK (understand_minutes >= cover_minutes),
  cover_bucket          TEXT NOT NULL,
  understand_bucket     TEXT NOT NULL,
  time_basis            TEXT NOT NULL,                   -- stated_in_post | estimated_from_post | post_only
  time_details          TEXT NOT NULL CHECK (json_valid(time_details)),
  alignment_score       REAL NOT NULL CHECK (alignment_score BETWEEN 0 AND 10),
  alignment_band        TEXT NOT NULL,
  alignment_subscores   TEXT NOT NULL CHECK (json_valid(alignment_subscores)),
  summary               TEXT NOT NULL,
  key_points            TEXT NOT NULL CHECK (json_valid(key_points)),
  rationales            TEXT NOT NULL CHECK (json_valid(rationales)),
  confidence            REAL NOT NULL,                   -- recomputed at load time (see the top of this file)
  confidence_parts      TEXT NOT NULL CHECK (json_valid(confidence_parts)),
  needs_review          BOOLEAN NOT NULL,                -- recomputed at load time with the current rule
  review_reasons        TEXT NOT NULL CHECK (json_valid(review_reasons)),
  review_rule_version   TEXT NOT NULL,
  review_rule_sha       TEXT NOT NULL,
  confidence_as_run     REAL NOT NULL,                   -- what the run itself wrote, kept for audit
  needs_review_as_run   BOOLEAN NOT NULL,
  clustering_subtopic_id TEXT,
  warnings              TEXT NOT NULL CHECK (json_valid(warnings)),
  attempts              TEXT NOT NULL CHECK (json_valid(attempts)),
  retry_reasons         TEXT NOT NULL CHECK (json_valid(retry_reasons)),
  is_active             BOOLEAN NOT NULL DEFAULT 1,
  model                 TEXT NOT NULL,
  prompt_version        TEXT NOT NULL,
  prompt_sha            TEXT NOT NULL,
  rubrics_version       TEXT NOT NULL,
  rubrics_sha           TEXT NOT NULL,
  taxonomy_version      TEXT NOT NULL REFERENCES taxonomy_versions(version),
  cache_key             TEXT NOT NULL UNIQUE,
  run_id                TEXT REFERENCES runs(id),
  input_tokens          INTEGER,
  output_tokens         INTEGER,
  seconds               REAL,
  created_at            TIMESTAMP NOT NULL
);
CREATE UNIQUE INDEX ux_active_cls ON classifications(item_id) WHERE is_active = 1;
CREATE INDEX ix_cls_topic ON classifications(topic_id, subtopic_id);

CREATE TABLE reviews (
  id                TEXT PRIMARY KEY,
  item_id           TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  classification_id TEXT REFERENCES classifications(id),
  corrected_fields  TEXT NOT NULL CHECK (json_valid(corrected_fields)),
  note              TEXT,
  use_as_example    BOOLEAN NOT NULL DEFAULT 1,
  created_at        TIMESTAMP NOT NULL
);

CREATE VIEW library AS
SELECT i.id, i.activity_id, i.title, i.author, i.kind, i.saved_at, i.post_date,
       i.post_url, i.resource_link, i.resource_links, i.resource_link_count,
       c.id AS classification_id,
       c.topic_id, t.name AS topic_name, c.subtopic_id, s.name AS subtopic_name,
       c.difficulty, c.content_type,
       c.cover_minutes, c.understand_minutes, c.cover_bucket, c.understand_bucket, c.time_basis,
       c.alignment_score, c.alignment_band, c.summary,
       c.confidence, c.needs_review, c.review_reasons, c.model
FROM items i
JOIN classifications c ON c.item_id = i.id AND c.is_active = 1
LEFT JOIN topics t ON t.taxonomy_version = c.taxonomy_version AND t.id = c.topic_id
LEFT JOIN subtopics s ON s.taxonomy_version = c.taxonomy_version AND s.id = c.subtopic_id
WHERE i.duplicate_of IS NULL;
"""


class LoadError(Exception):
    """The inputs have problems; the message lists all of them."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ the review rule, re-applied

def rescore(record: dict, taxonomy: I.Taxonomy, rule: RV.ReviewRule) -> dict:
    """Confidence and review fields of one stored record, worked out again from its stored parts.

    Same arithmetic as the run (classifier.compute_confidence); the only difference is that a
    post the clustering could not place gives "no opinion" instead of "disagrees" (item 54).
    """
    parts = record["confidence_parts"]
    clustering = record.get("clustering_subtopic_id")
    if clustering == I.OTHER_ID:          # (a post with no clustering at all gives None in C.agreement)
        agree = None
    else:
        agree = C.agreement(record["subtopic_id"], clustering, taxonomy)
    first_try = parts["valid"] >= C.CONFIDENCE["valid_first_try"]
    confidence, new_parts = C.compute_confidence(
        self_confidence=parts["self"], agree=agree, first_try=first_try,
        partial=bool(parts["partial"]), penalties=int(parts["penalties"]))
    reasons = RV.review_reasons(rule, confidence=confidence, subtopic_id=record["subtopic_id"],
                                clustering_subtopic_id=clustering, taxonomy=taxonomy)
    return {"confidence": confidence, "confidence_parts": new_parts,
            "review_reasons": reasons, "needs_review": bool(reasons)}


# ------------------------------------------------------------------ reading the inputs

def split_links(urls: str) -> list[str]:
    """The `urls` cell of the posts CSV -> a list of links (empty cell -> empty list)."""
    return [u.strip() for u in (urls or "").split(RESOURCE_SEPARATOR.strip()) if u.strip()]


def make_title(attachment_title: str, text: str) -> str:
    """A post has no title: use the attachment's title, else the first line of the text, cut short."""
    attachment_title = (attachment_title or "").strip()
    if attachment_title:
        return attachment_title[:TITLE_CHARS]
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= TITLE_CHARS else line[:TITLE_CHARS - 1].rstrip() + "…"
    return ""


def read_posts_csv(path: Path) -> dict[int, dict]:
    """activity_id -> the CSV row. Every row must have a post link (the dashboard links every result)."""
    problems: list[str] = []
    rows: dict[int, dict] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            missing = {"activity_id", "post_url"} - set(reader.fieldnames or [])
            if missing:
                raise LoadError(f"posts CSV problems:\n- missing columns {sorted(missing)}")
            for line_no, row in enumerate(reader, start=2):
                try:
                    aid = int((row["activity_id"] or "").strip())
                except ValueError:
                    problems.append(f"line {line_no}: activity_id '{row['activity_id']}' is not a whole number")
                    continue
                if aid in rows:
                    problems.append(f"line {line_no}: activity_id {aid} appears twice")
                    continue
                url = (row.get("post_url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    problems.append(f"line {line_no}: activity_id {aid} has no usable post_url ('{url[:40]}')")
                    continue
                rows[aid] = row
    except OSError as exc:
        raise LoadError(f"posts CSV problems:\n- cannot read {path}: {exc.strerror or exc}") from exc
    if problems:
        raise LoadError("posts CSV problems:\n- " + "\n- ".join(problems))
    return rows


def read_records(path: Path) -> tuple[list[dict], int, str]:
    """The records of one .jsonl (the last line wins when a post appears twice).

    Returns (records, number of superseded duplicate lines, sha256 of the file).
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoadError(f"results problems:\n- cannot read {path}: {exc.strerror or exc}") from exc
    by_id: dict[int, dict] = {}
    problems: list[str] = []
    duplicates = 0
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("activity_id"), int):
            problems.append(f"line {line_no}: no whole-number activity_id")
            continue
        if "error" in rec and "subtopic_id" not in rec:       # a failure line, not a result
            continue
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            problems.append(f"line {line_no}: activity_id {rec['activity_id']} lacks fields {missing}")
            continue
        if rec["activity_id"] in by_id:
            duplicates += 1
        by_id[rec["activity_id"]] = rec
    if problems:
        raise LoadError("results problems:\n- " + "\n- ".join(problems))
    return list(by_id.values()), duplicates, hashlib.sha256(raw).hexdigest()


REQUIRED_FIELDS = (
    "activity_id", "topic_id", "subtopic_id", "difficulty", "content_type", "time", "alignment_score",
    "alignment_band", "alignment_subscores", "summary", "key_points", "rationales", "confidence",
    "confidence_parts", "needs_review", "clustering_subtopic_id", "warnings", "model", "prompt_version",
    "prompt_sha", "taxonomy_version", "rubrics_version", "rubrics_sha", "cache_key", "attempts",
    "retry_reasons", "created_at")


def check_records(records: list[dict], posts: dict[int, dict], taxonomy: I.Taxonomy) -> list[str]:
    problems = []
    valid = set(taxonomy.choice_ids)
    for rec in records:
        aid = rec["activity_id"]
        if aid not in posts:
            problems.append(f"activity_id {aid} is in the results but not in the posts CSV")
        if rec["taxonomy_version"] != taxonomy.version:
            problems.append(f"activity_id {aid}: classified with taxonomy '{rec['taxonomy_version']}' "
                            f"but the taxonomy file is '{taxonomy.version}'")
        if rec["subtopic_id"] not in valid:
            problems.append(f"activity_id {aid}: subtopic '{rec['subtopic_id']}' is not in the taxonomy")
    return problems[:50] + ([f"... and {len(problems) - 50} more"] if len(problems) > 50 else [])


# ------------------------------------------------------------------ writing the database

def open_db(path: Path) -> sqlite3.Connection:
    """Open (and create if missing) the database. Refuses a file that is not ours."""
    new = not path.exists()
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if new or (version == 0 and not con.execute("SELECT 1 FROM sqlite_master").fetchone()):
        con.executescript(SCHEMA)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        con.commit()
    elif version != SCHEMA_VERSION:
        con.close()
        raise LoadError(f"{path} is not a database made by this loader (schema version {version}, "
                        f"expected {SCHEMA_VERSION})")
    return con


def _taxonomy_tree(taxonomy: I.Taxonomy) -> dict:
    return yaml.safe_load(taxonomy.path.read_text(encoding="utf-8"))


def _load_taxonomy(con: sqlite3.Connection, taxonomy: I.Taxonomy, now: str) -> None:
    row = con.execute("SELECT sha256 FROM taxonomy_versions WHERE version = ?", (taxonomy.version,)).fetchone()
    if row is not None:
        if row["sha256"] != taxonomy.sha:
            raise LoadError(f"taxonomy '{taxonomy.version}' is already in the database with different "
                            f"content; give the changed taxonomy a new version name")
        return
    tree = _taxonomy_tree(taxonomy)
    con.execute("INSERT INTO taxonomy_versions(version, sha256, tree, created_at) VALUES (?,?,?,?)",
                (taxonomy.version, taxonomy.sha, _js(tree), now))
    topic_names = {s.topic_id: s.topic_name for s in taxonomy.subtopics}
    other = tree["other"]
    topic_names[I.OTHER_TOPIC_ID] = other["name"].strip()
    con.executemany("INSERT INTO topics(taxonomy_version, id, name) VALUES (?,?,?)",
                    [(taxonomy.version, tid, name) for tid, name in topic_names.items()])
    rows = [(taxonomy.version, s.id, s.topic_id, s.name) for s in taxonomy.subtopics]
    rows.append((taxonomy.version, I.OTHER_ID, I.OTHER_TOPIC_ID, other["name"].strip()))
    con.executemany("INSERT INTO subtopics(taxonomy_version, id, topic_id, name) VALUES (?,?,?,?)", rows)


def _load_item(con: sqlite3.Connection, aid: int, row: dict, has_result: bool, partial: bool, now: str) -> None:
    text = (row.get("text_clean") or "").strip() or (row.get("text") or "").strip()
    links = split_links(row.get("urls") or "")
    status = CLASSIFIED if has_result else NO_TEXT
    con.execute(
        """INSERT INTO items(id, activity_id, post_url, resource_links, resource_link, resource_link_count,
                             kind, collector, author, title, post_date, saved_at, capture_status, status,
                             created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             post_url=excluded.post_url, resource_links=excluded.resource_links,
             resource_link=excluded.resource_link, resource_link_count=excluded.resource_link_count,
             collector=excluded.collector, author=excluded.author, title=excluded.title,
             post_date=excluded.post_date, saved_at=excluded.saved_at,
             capture_status=excluded.capture_status, status=excluded.status, updated_at=excluded.updated_at""",
        (str(aid), aid, row["post_url"].strip(), _js(links), links[0] if links else None, len(links),
         KIND, (row.get("source_file") or "").strip() or UNKNOWN_COLLECTOR,
         (row.get("author") or "").strip() or None,
         make_title(row.get("attachment_title") or "", text) or None,
         (row.get("post_date_utc") or "").strip() or None, (row.get("saved_at") or "").strip() or None,
         (row.get("status") or "").strip() or "captured", status, now, now))
    if not text:
        con.execute("DELETE FROM contents WHERE item_id = ?", (str(aid),))
        return
    metrics = {"word_count": len(text.split()), "char_count": len(text),
               "has_image": (row.get("has_image") or "").strip().lower() == "true",
               "url_count": len(links), "attachment_title": (row.get("attachment_title") or "").strip(),
               "attachment_site": (row.get("attachment_site") or "").strip()}
    con.execute(
        """INSERT INTO contents(item_id, body_text, content_hash, metrics, partial) VALUES (?,?,?,?,?)
           ON CONFLICT(item_id) DO UPDATE SET body_text=excluded.body_text,
             content_hash=excluded.content_hash, metrics=excluded.metrics, partial=excluded.partial""",
        (str(aid), text, hashlib.sha256(text.encode("utf-8")).hexdigest(), _js(metrics), int(partial)))


def _load_run(con: sqlite3.Connection, run_id: str, path: Path, file_sha: str, records: list[dict],
              rule: RV.ReviewRule) -> None:
    first = records[0]
    stamps = sorted(r["created_at"] for r in records)
    snapshot = {"file": path.name, "file_sha256": file_sha, "model": first["model"],
                "prompt_version": first["prompt_version"], "prompt_sha": first["prompt_sha"],
                "rubrics_version": first["rubrics_version"], "rubrics_sha": first["rubrics_sha"],
                "taxonomy_version": first["taxonomy_version"],
                "review_rule_as_run": {"version": first.get("review_rule_version"), "sha": first.get("review_rule_sha")},
                "review_rule_at_load": {"version": rule.version, "sha": rule.sha,
                                        "confidence_below": rule.confidence_below}}
    con.execute(
        """INSERT INTO runs(id, command, config_snapshot, items_ok, items_failed, input_tokens, output_tokens,
                            est_cost, started_at, finished_at)
           VALUES (?,?,?,?,?,?,?,0,?,?)
           ON CONFLICT(id) DO UPDATE SET command=excluded.command, config_snapshot=excluded.config_snapshot,
             items_ok=excluded.items_ok, input_tokens=excluded.input_tokens,
             output_tokens=excluded.output_tokens, started_at=excluded.started_at,
             finished_at=excluded.finished_at""",
        (run_id, f"loaded from {path.name} by step15_load_sqlite.py", _js(snapshot), len(records), 0,
         sum(r.get("input_tokens") or 0 for r in records), sum(r.get("output_tokens") or 0 for r in records),
         stamps[0], stamps[-1]))


def _classification_row(rec: dict, new: dict, run_id: str) -> tuple:
    t = rec["time"]
    return (
        "cls_" + rec["cache_key"][:16], str(rec["activity_id"]), rec["topic_id"], rec["subtopic_id"],
        rec["difficulty"], rec["content_type"], t["cover_minutes"], t["understand_minutes"],
        t["cover_bucket"], t["understand_bucket"], t["basis"], _js(t),
        rec["alignment_score"], rec["alignment_band"], _js(rec["alignment_subscores"]),
        rec["summary"], _js(rec["key_points"]), _js(rec["rationales"]),
        new["confidence"], _js(new["confidence_parts"]), int(new["needs_review"]), _js(new["review_reasons"]),
        new["rule_version"], new["rule_sha"], rec["confidence"], int(bool(rec["needs_review"])),
        rec["clustering_subtopic_id"], _js(rec["warnings"]), _js(rec["attempts"]), _js(rec["retry_reasons"]),
        rec["model"], rec["prompt_version"], rec["prompt_sha"], rec["rubrics_version"], rec["rubrics_sha"],
        rec["taxonomy_version"], rec["cache_key"], run_id,
        rec.get("input_tokens"), rec.get("output_tokens"), rec.get("seconds"), rec["created_at"])


INSERT_CLASSIFICATION = """INSERT INTO classifications(
  id, item_id, topic_id, subtopic_id, difficulty, content_type, cover_minutes, understand_minutes,
  cover_bucket, understand_bucket, time_basis, time_details, alignment_score, alignment_band,
  alignment_subscores, summary, key_points, rationales, confidence, confidence_parts, needs_review,
  review_reasons, review_rule_version, review_rule_sha, confidence_as_run, needs_review_as_run,
  clustering_subtopic_id, warnings, attempts, retry_reasons, model, prompt_version, prompt_sha,
  rubrics_version, rubrics_sha, taxonomy_version, cache_key, run_id, input_tokens, output_tokens,
  seconds, created_at, is_active)
  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)"""


@dataclass
class LoadReport:
    items: int = 0
    items_without_text: int = 0
    records: int = 0
    inserted: int = 0
    refreshed: int = 0
    superseded: int = 0                  # older active rows switched off by a newer run
    duplicate_lines: int = 0
    needs_review_as_run: int = 0
    needs_review_now: int = 0
    noise_posts: int = 0
    noise_flagged_as_run: int = 0
    noise_flagged_now: int = 0
    resource_link_posts: int = 0
    run_id: str = ""
    problems: list[str] = field(default_factory=list)


def load(db_path: Path | str, jsonl_path: Path | str, csv_path: Path | str, taxonomy_path: Path | str,
         review_rule_path: Path | str = RV.DEFAULT_PATH) -> LoadReport:
    """Read the files, check them all, then write everything in one transaction."""
    db_path, jsonl_path, csv_path = Path(db_path), Path(jsonl_path), Path(csv_path)
    try:
        taxonomy = I.load_taxonomy(taxonomy_path)
        rule = RV.load_review_rule(review_rule_path)
    except (I.InputError, RV.ReviewRuleError) as exc:
        raise LoadError(str(exc)) from exc
    posts = read_posts_csv(csv_path)
    records, duplicates, file_sha = read_records(jsonl_path)
    if not records:
        raise LoadError(f"results problems:\n- {jsonl_path} has no classified posts")
    problems = check_records(records, posts, taxonomy)
    if problems:
        raise LoadError("results problems:\n- " + "\n- ".join(problems))

    report = LoadReport(duplicate_lines=duplicates, records=len(records), run_id="run_" + jsonl_path.stem)
    rescored = {r["activity_id"]: rescore(r, taxonomy, rule) for r in records}
    for rec in records:
        new = rescored[rec["activity_id"]]
        new["rule_version"], new["rule_sha"] = rule.version, rule.sha
        noise = rec["clustering_subtopic_id"] == I.OTHER_ID
        report.needs_review_as_run += bool(rec["needs_review"])
        report.needs_review_now += new["needs_review"]
        report.noise_posts += noise
        report.noise_flagged_as_run += noise and bool(rec["needs_review"])
        report.noise_flagged_now += noise and new["needs_review"]

    now = utc_now()
    con = open_db(db_path)
    try:
        with con:                                            # one transaction: all or nothing
            _load_taxonomy(con, taxonomy, now)
            by_id = {r["activity_id"]: r for r in records}
            for aid, row in posts.items():
                rec = by_id.get(aid)
                partial = bool(rec) and "text_partial" in rec["warnings"]
                _load_item(con, aid, row, rec is not None, partial, now)
                report.items += 1
                report.items_without_text += rec is None
                report.resource_link_posts += bool(split_links(row.get("urls") or ""))
            _load_run(con, report.run_id, jsonl_path, file_sha, records, rule)
            for rec in records:
                new = rescored[rec["activity_id"]]
                new["rule_version"], new["rule_sha"] = rule.version, rule.sha
                existing = con.execute("SELECT id FROM classifications WHERE cache_key = ?",
                                       (rec["cache_key"],)).fetchone()
                if existing:
                    con.execute(
                        """UPDATE classifications SET confidence=?, confidence_parts=?, needs_review=?,
                           review_reasons=?, review_rule_version=?, review_rule_sha=? WHERE id=?""",
                        (new["confidence"], _js(new["confidence_parts"]), int(new["needs_review"]),
                         _js(new["review_reasons"]), rule.version, rule.sha, existing["id"]))
                    report.refreshed += 1
                    continue
                switched = con.execute("UPDATE classifications SET is_active = 0 WHERE item_id = ? AND is_active = 1",
                                       (str(rec["activity_id"]),)).rowcount
                report.superseded += switched
                con.execute(INSERT_CLASSIFICATION, _classification_row(rec, new, report.run_id))
                report.inserted += 1
        report.problems = verify(con, posts)
    except sqlite3.Error as exc:                              # e.g. a value the table rules refuse; nothing was saved
        raise LoadError(f"database problems (nothing was saved):\n- {exc}") from exc
    finally:
        con.close()
    return report


# ------------------------------------------------------------------ checks on the finished database

def verify(con: sqlite3.Connection, posts: dict[int, dict] | None = None) -> list[str]:
    """Things that must be true after a load. Returns the problems (empty list = all fine)."""
    problems = []
    n = con.execute("SELECT COUNT(*) FROM items WHERE post_url IS NULL OR post_url = ''").fetchone()[0]
    if n:
        problems.append(f"{n} items have no post link")
    n = con.execute("SELECT COUNT(*) FROM library WHERE post_url IS NULL OR post_url = ''").fetchone()[0]
    if n:
        problems.append(f"{n} library rows have no post link")
    n = con.execute("""SELECT COUNT(*) FROM items WHERE resource_link_count > 0
                       AND (resource_link IS NULL OR resource_links = '[]')""").fetchone()[0]
    if n:
        problems.append(f"{n} items count resource links but have none stored")
    n = con.execute("""SELECT COUNT(*) FROM items i WHERE status = 'classified' AND NOT EXISTS
                       (SELECT 1 FROM classifications c WHERE c.item_id = i.id AND c.is_active = 1)""").fetchone()[0]
    if n:
        problems.append(f"{n} classified items have no active classification")
    n = con.execute("""SELECT COUNT(*) FROM (SELECT item_id FROM classifications WHERE is_active = 1
                       GROUP BY item_id HAVING COUNT(*) > 1)""").fetchone()[0]
    if n:
        problems.append(f"{n} items have more than one active classification")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    if bad:
        problems.append(f"{len(bad)} rows point at something that does not exist")
    if posts is not None:
        with_links = sum(1 for r in posts.values() if split_links(r.get("urls") or ""))
        kept = con.execute("SELECT COUNT(*) FROM items WHERE resource_link_count > 0").fetchone()[0]
        if kept != with_links:
            problems.append(f"the posts CSV has {with_links} posts with resource links, the database {kept}")
    return problems


# ------------------------------------------------------------------ command line

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load classified posts into a SQLite database (Step 15).")
    p.add_argument("--jsonl", required=True, help="the results file of step9_classify.py")
    p.add_argument("--csv", required=True, help="posts_with_save_dates.csv (the source of the links)")
    p.add_argument("--taxonomy", required=True, help="taxonomy.v1.yaml")
    p.add_argument("--db", default="data/library.sqlite", help="database file (created if missing)")
    p.add_argument("--review-rule", default=str(RV.DEFAULT_PATH), help="review rule file (default: v1)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rep = load(args.db, args.jsonl, args.csv, args.taxonomy, args.review_rule)
    except LoadError as exc:
        print(f"CANNOT LOAD:\n{exc}")
        return 2
    print(f"Loaded {args.jsonl} into {args.db} (run {rep.run_id})")
    print(f"  items: {rep.items} ({rep.items_without_text} without text, so not classified)")
    print(f"  classifications: {rep.inserted} new, {rep.refreshed} refreshed, {rep.superseded} older ones switched off")
    if rep.duplicate_lines:
        print(f"  {rep.duplicate_lines} repeated lines in the results file (the last one of each post was used)")
    print(f"  posts with resource links: {rep.resource_link_posts}")
    print(f"  needs review: {rep.needs_review_now} of {rep.records} now, {rep.needs_review_as_run} as the run wrote it")
    print(f"  clustering-noise posts: {rep.noise_posts}; flagged now {rep.noise_flagged_now}, "
          f"as the run wrote it {rep.noise_flagged_as_run}")
    if rep.problems:
        print("CHECKS FAILED:")
        for p in rep.problems:
            print(f"  - {p}")
        return 1
    print("  checks passed (every item has a post link, resource links kept, one active row per post)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
