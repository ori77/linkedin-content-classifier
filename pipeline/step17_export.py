"""Step 17: export the library and your corrections to CSV and JSON.

In simple terms: this turns the database into files you can open in Excel, load with pandas or
keep as a backup. Two kinds of file:

  library      one row per classified post: its labels (YOUR corrected label where you fixed
               one), times, score, summary, confidence, review status and the links.
  corrections  one row per post you reviewed (approved or fixed): what the model said, what you
               decided, your note, and whether to use it as an example later (chat 4).

EVERY row of EVERY file carries the links back to the source: `post_url` (the LinkedIn post),
`resource_link` (the first link found in the post, or empty), `resource_links` (all of them) and
`resource_link_count`. Links are only written down here; nothing opens or fetches them.

Run:  python pipeline/step17_export.py [--db data/library.sqlite] [--out-dir exports]
                                       [--what library|corrections|both] [--format csv|json|both]
Files written:  library.csv  library.json  corrections.csv  corrections.json  (in --out-dir)
Exit code: 0 ok, 2 cannot export (no database, folder cannot be written).
The dashboard's download buttons use the same functions, so a download matches a file.

Things to know when you open the files:
  * CSV is UTF-8 with a byte-order mark, so Excel shows accents and emoji correctly.
  * In CSV, a text that starts with = + - @ gets a leading ' (otherwise Excel may run it as a
    formula). JSON is never changed.
  * Lists are joined with " | " in CSV and are real lists in JSON; true/false is written as
    true/false; empty means "none".
  * `activity_id` is written as text in JSON (the number is too big for some tools). Excel cuts
    long numbers in CSV, so use `post_url` there: it has the full id.
  * Exports hold your saved posts' titles, summaries and links: /exports/ is in .gitignore.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

import dashboard_data as D

DEFAULT_OUT_DIR = "exports"
WHATS = ("library", "corrections", "both")
FORMATS = ("csv", "json", "both")
LIST_SEPARATOR = " | "
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")

LINK_COLUMNS = ("post_url", "resource_link", "resource_links", "resource_link_count")

LIBRARY_COLUMNS = (
    "activity_id", "title", "author", "saved_at", "post_date",
    "post_url", "resource_link", "resource_links", "resource_link_count",
    "topic_id", "topic_name", "subtopic_id", "subtopic_name", "difficulty", "content_type",
    "cover_minutes", "cover_bucket", "understand_minutes", "understand_bucket", "time_basis",
    "alignment_score", "alignment_band", "summary", "confidence",
    "status", "model_flagged", "review_reasons", "reviewed", "changed_fields", "review_note",
    "model_subtopic_id", "model_difficulty", "model_content_type", "model", "taxonomy_version",
)

CORRECTION_COLUMNS = (
    "activity_id", "title", "author",
    "post_url", "resource_link", "resource_links", "resource_link_count",
    "review_id", "reviewed_at", "decision", "changed_fields", "note", "use_as_example",
    "model_subtopic_id", "model_difficulty", "model_content_type",
    "final_subtopic_id", "final_difficulty", "final_content_type",
    "summary", "model_flagged", "review_reasons", "confidence", "model", "taxonomy_version",
)


class ExportError(Exception):
    """Something the script can explain in one sentence."""


# ------------------------------------------------------------------ the records

def _links(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"post_url": row["post_url"], "resource_link": row["resource_link"],
            "resource_links": list(row["resource_links_list"]),
            "resource_link_count": row["resource_link_count"]}


def library_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per post, in the order given. Labels are the ones that count now (your fix wins)."""
    out = []
    for r in rows:
        rec = {
            "activity_id": str(r["activity_id"]), "title": r["title"], "author": r["author"],
            "saved_at": r["saved_at"], "post_date": r["post_date"],
            **_links(r),
            "topic_id": r["topic_id"], "topic_name": r["topic_name"],
            "subtopic_id": r["subtopic_id"], "subtopic_name": r["subtopic_name"],
            "difficulty": r["difficulty"], "content_type": r["content_type"],
            "cover_minutes": r["cover_minutes"], "cover_bucket": r["cover_bucket"],
            "understand_minutes": r["understand_minutes"], "understand_bucket": r["understand_bucket"],
            "time_basis": r["time_basis"],
            "alignment_score": r["alignment_score"], "alignment_band": r["alignment_band"],
            "summary": r["summary"], "confidence": r["confidence"],
            "status": r["status"], "model_flagged": r["model_flagged"],
            "review_reasons": list(r["review_reasons"]), "reviewed": r["reviewed"],
            "changed_fields": list(r["corrected_fields"]), "review_note": r["review_note"],
            "model_subtopic_id": r["model_subtopic_id"], "model_difficulty": r["model_difficulty"],
            "model_content_type": r["model_content_type"], "model": r["model"],
            "taxonomy_version": r["taxonomy_version"],
        }
        out.append(rec)
    return out


def correction_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per REVIEWED post (its newest review), oldest review first."""
    reviewed = sorted((r for r in rows if r["reviewed"]), key=lambda r: (r["reviewed_at"], r["activity_id"]))
    out = []
    for r in reviewed:
        rec = {
            "activity_id": str(r["activity_id"]), "title": r["title"], "author": r["author"],
            **_links(r),
            "review_id": r["review_id"], "reviewed_at": r["reviewed_at"],
            "decision": "corrected" if r["corrected_fields"] else "approved",
            "changed_fields": list(r["corrected_fields"]), "note": r["review_note"],
            "use_as_example": r["use_as_example"],
            "model_subtopic_id": r["model_subtopic_id"], "model_difficulty": r["model_difficulty"],
            "model_content_type": r["model_content_type"],
            "final_subtopic_id": r["subtopic_id"], "final_difficulty": r["difficulty"],
            "final_content_type": r["content_type"],
            "summary": r["summary"], "model_flagged": r["model_flagged"],
            "review_reasons": list(r["review_reasons"]), "confidence": r["confidence"],
            "model": r["model"], "taxonomy_version": r["taxonomy_version"],
        }
        out.append(rec)
    return out


# ------------------------------------------------------------------ the two file formats

def csv_cell(value: Any) -> str:
    """One value as CSV text. Text that could run as a spreadsheet formula gets a leading '."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)                                 # a number is never a formula, even a negative one
    if isinstance(value, list):
        text = LIST_SEPARATOR.join(str(v) for v in value)
    else:
        text = str(value)
    if text.startswith(FORMULA_STARTS):
        text = "'" + text
    return text


def to_csv_bytes(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)                     # quotes commas, quotes and line breaks itself
    writer.writerow(columns)
    for rec in records:
        writer.writerow([csv_cell(rec[c]) for c in columns])
    return buf.getvalue().encode("utf-8-sig")


def to_json_bytes(records: Sequence[Mapping[str, Any]], kind: str, exported_at: str) -> bytes:
    doc = {"kind": kind, "exported_at": exported_at, "count": len(records), "rows": list(records)}
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# ------------------------------------------------------------------ writing files

def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary file next to the target, then swap it in, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def export_files(rows: Sequence[Mapping[str, Any]], out_dir: Path | str, what: str = "both",
                 fmt: str = "both", exported_at: str | None = None) -> list[tuple[Path, int]]:
    """Write the chosen files. Returns [(path, number of rows)]."""
    if what not in WHATS:
        raise ExportError(f"--what must be one of {list(WHATS)}, not '{what}'")
    if fmt not in FORMATS:
        raise ExportError(f"--format must be one of {list(FORMATS)}, not '{fmt}'")
    stamp = exported_at or D.utc_now()
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"cannot create the folder {out_dir}: {exc}") from exc
    plan = []
    if what in ("library", "both"):
        plan.append(("library", library_records(rows), LIBRARY_COLUMNS))
    if what in ("corrections", "both"):
        plan.append(("corrections", correction_records(rows), CORRECTION_COLUMNS))
    written = []
    for kind, records, columns in plan:
        try:
            if fmt in ("csv", "both"):
                path = out_dir / f"{kind}.csv"
                write_atomic(path, to_csv_bytes(records, columns))
                written.append((path, len(records)))
            if fmt in ("json", "both"):
                path = out_dir / f"{kind}.json"
                write_atomic(path, to_json_bytes(records, kind, stamp))
                written.append((path, len(records)))
        except OSError as exc:
            raise ExportError(f"cannot write into {out_dir}: {exc}") from exc
    return written


# ------------------------------------------------------------------ the dashboard's download buttons

def _downloads(records: Sequence[Mapping[str, Any]], kind: str, columns: Sequence[str], key_prefix: str,
               phrase: str, exported_at: str | None = None) -> list[dict[str, Any]]:
    """What the screen needs for a CSV button and a JSON button: key, label, data, file name, type, disabled."""
    n = len(records)
    name = phrase.format(n=n)
    stamp = exported_at or D.utc_now()
    return [
        {"key": f"{key_prefix}_dl_csv", "label": f"⬇ {name} as CSV", "data": to_csv_bytes(records, columns),
         "file_name": f"{kind}.csv", "mime": "text/csv", "disabled": n == 0},
        {"key": f"{key_prefix}_dl_json", "label": f"⬇ {name} as JSON", "data": to_json_bytes(records, kind, stamp),
         "file_name": f"{kind}.json", "mime": "application/json", "disabled": n == 0},
    ]


def library_downloads(shown: Sequence[Mapping[str, Any]], exported_at: str | None = None) -> list[dict[str, Any]]:
    """The Library tab's buttons: exactly the posts the filters show now."""
    return _downloads(library_records(shown), "library", LIBRARY_COLUMNS, "lib", "These {n} posts", exported_at)


def correction_downloads(rows: Sequence[Mapping[str, Any]], exported_at: str | None = None) -> list[dict[str, Any]]:
    """The Review tab's buttons: all your reviews, whatever the Library filters say."""
    return _downloads(correction_records(rows), "corrections", CORRECTION_COLUMNS, "rv", "My {n} reviews", exported_at)


# ------------------------------------------------------------------ command line

def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Export the library and your corrections to CSV and JSON.")
    parser.add_argument("--db", default=env.get("LISC_DB") or D.DEFAULT_DB, help="the database (default data/library.sqlite)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="folder for the files (default exports)")
    parser.add_argument("--what", choices=WHATS, default="both")
    parser.add_argument("--format", choices=FORMATS, default="both", dest="fmt")
    args = parser.parse_args(argv)
    try:
        with closing(D.connect(args.db)) as con:
            rows = D.load_library(con)
        written = export_files(rows, args.out_dir, args.what, args.fmt)
    except (D.DashboardError, ExportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for path, n in written:
        print(f"wrote {path}  ({n} rows)")
    reviewed = sum(1 for r in rows if r["reviewed"])
    print(f"{len(rows)} posts in the library, {reviewed} reviewed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
