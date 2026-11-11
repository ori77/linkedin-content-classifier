"""Step 10b: read the labelled gold sheet, check it, and prepare the classifier run on those posts.

In simple terms
---------------
You filled in gold_labelling.xlsx by hand. This script:
  1. reads the sheet (only the values; Excel's extras are ignored),
  2. checks every filled cell (the subtopic must be on the list, the type and difficulty must be
     valid, alignment 0 to 10, minutes whole numbers) and lists ALL problems together,
  3. accepts blanks: a blank cell just means "no opinion", and later comparisons skip it,
  4. writes gold/gold_labels.csv (clean labels, one row per post) and gold/gold_ids.txt (the post ids,
     comma-separated, to paste after --ids when running the classifier on exactly these posts).

If the key file is given, it also checks that the ids in the sheet are the ids that were sampled.
Nothing in the sheet is changed.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import classify_inputs as I
import rubrics as R
import step10_gold_sample as G

SUBTOPIC_SEP = " | "
CORE = ("my_subtopic", "my_content_type", "my_difficulty")          # the three main labels
OPTIONAL = ("my_alignment", "my_cover_minutes", "my_understand_minutes", "notes")
LABEL_FILE = "gold_labels.csv"
IDS_FILE = "gold_ids.txt"
CSV_COLUMNS = ["gold_id", "activity_id", "subtopic_id", "content_type", "difficulty",
               "alignment", "cover_minutes", "understand_minutes", "notes"]


class LabelError(Exception):
    """The labelled sheet has problems; the message lists all of them."""


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _number(v: Any) -> float | None:
    """A real number from a cell, or None. Booleans are not numbers; text such as '7' is read as 7."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def clean_row(row: dict, tax: I.Taxonomy, where: str, problems: list[str]) -> dict:
    """Check one sheet row and return its clean labels (blank = None). Problems are appended, not raised."""
    out: dict[str, Any] = {"subtopic_id": None, "content_type": None, "difficulty": None, "alignment": None,
                           "cover_minutes": None, "understand_minutes": None, "notes": None}
    v = row.get("my_subtopic")
    if not _blank(v):
        sid = str(v).split(SUBTOPIC_SEP)[0].strip()
        if sid in tax.choice_ids:
            out["subtopic_id"] = sid
        else:
            problems.append(f"{where}: my_subtopic '{v}' is not on the list")
    for col, key, valid in (("my_content_type", "content_type", R.CONTENT_TYPE_IDS),
                            ("my_difficulty", "difficulty", R.DIFFICULTY_IDS)):
        v = row.get(col)
        if not _blank(v):
            if str(v).strip() in valid:
                out[key] = str(v).strip()
            else:
                problems.append(f"{where}: {col} '{v}' must be one of {', '.join(valid)}")
    v = row.get("my_alignment")
    if not _blank(v):
        n = _number(v)
        if n is None or not 0 <= n <= 10:
            problems.append(f"{where}: my_alignment '{v}' must be a number from 0 to 10")
        else:
            out["alignment"] = n
    for col, key in (("my_cover_minutes", "cover_minutes"), ("my_understand_minutes", "understand_minutes")):
        v = row.get(col)
        if not _blank(v):
            n = _number(v)
            if n is None or n < 1 or n != int(n):
                problems.append(f"{where}: {col} '{v}' must be a whole number of minutes, 1 or more")
            else:
                out[key] = int(n)
    if not _blank(row.get("notes")):
        out["notes"] = " ".join(str(row["notes"]).split())
    return out


def read_labels(xlsx: Path | str, tax: I.Taxonomy, key_ids: dict[str, int] | None = None) -> list[dict]:
    """Rows of clean labels in sheet order. Raises LabelError listing every problem."""
    import warnings
    import zipfile

    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    try:
        with warnings.catch_warnings():
            # "Data Validation extension is not supported" is harmless. openpyxl raises it while the rows are
            # being read (not only at load time), so both steps sit inside the same quiet block.
            warnings.simplefilter("ignore")
            wb = load_workbook(xlsx, read_only=True, data_only=True)
            if G.SHEET_NAME not in wb.sheetnames:
                raise LabelError(f"labels problems:\n- the workbook has no '{G.SHEET_NAME}' sheet")
            rows = list(wb[G.SHEET_NAME].iter_rows(values_only=True))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, InvalidFileException) as exc:
        raise LabelError(f"labels problems:\n- cannot read {xlsx}: {exc}") from exc
    if not rows:
        raise LabelError("labels problems:\n- the sheet is empty")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    missing = [c for c in G.SHEET_COLUMNS if c not in header]
    if missing:
        raise LabelError(f"labels problems:\n- missing columns {missing}")
    problems: list[str] = []
    result: list[dict] = []
    seen: set[str] = set()
    for n, values in enumerate(rows[1:], start=2):
        row = dict(zip(header, values))
        if all(_blank(v) for v in values):
            continue                                        # an empty line at the bottom
        gid = str(row["gold_id"]).strip() if not _blank(row["gold_id"]) else ""
        where = f"row {n} ({gid or 'no gold_id'})"
        if not gid:
            problems.append(f"{where}: gold_id is empty")
        elif gid in seen:
            problems.append(f"{where}: gold_id appears twice")
        seen.add(gid)
        aid_raw = row["activity_id"]
        aid = None
        if isinstance(aid_raw, str) and aid_raw.strip().isdigit():
            aid = int(aid_raw.strip())
        else:
            problems.append(f"{where}: activity_id '{aid_raw}' is not a whole number kept as text "
                            "(Excel may have turned it into a rounded number)")
        if aid is not None and key_ids is not None:
            if gid not in key_ids:
                problems.append(f"{where}: gold_id is not in the key file")
            elif key_ids[gid] != aid:
                problems.append(f"{where}: activity_id {aid} differs from the key ({key_ids[gid]})")
        labels = clean_row(row, tax, where, problems)
        result.append({"gold_id": gid, "activity_id": aid, **labels})
    if key_ids is not None:
        for gid in sorted(set(key_ids) - seen):
            problems.append(f"{gid} is in the key file but not in the sheet")
    if problems:
        raise LabelError("labels problems:\n- " + "\n- ".join(problems))
    return result


def read_key(path: Path | str) -> dict[str, int]:
    out: dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["gold_id"]] = int(row["activity_id"])
    return out


def coverage(rows: list[dict]) -> dict[str, int]:
    """How many posts have each label filled."""
    return {k: sum(1 for r in rows if r[k] is not None)
            for k in ("subtopic_id", "content_type", "difficulty", "alignment", "cover_minutes", "understand_minutes")}


def write_outputs(out_dir: Path | str, rows: list[dict]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = out_dir / LABEL_FILE
    with labels.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in CSV_COLUMNS])
    ids = out_dir / IDS_FILE
    ids.write_text(",".join(str(r["activity_id"]) for r in rows) + "\n", encoding="utf-8")
    return labels, ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read and check the labelled gold sheet.")
    ap.add_argument("--sheet", required=True, help="the filled gold_labelling.xlsx")
    ap.add_argument("--taxonomy", required=True, help="taxonomy.v1.yaml")
    ap.add_argument("--key", help="gold_key.csv (optional; checks the ids match the sample)")
    ap.add_argument("--out-dir", default="gold")
    args = ap.parse_args(argv)
    try:
        tax = I.load_taxonomy(args.taxonomy)
        key = read_key(args.key) if args.key else None
        rows = read_labels(args.sheet, tax, key)
        labels, ids = write_outputs(args.out_dir, rows)
    except (I.InputError, LabelError, ValueError, KeyError, OSError) as exc:
        print(f"cannot read the labels: {exc}", file=sys.stderr)
        return 1
    cov = coverage(rows)
    print(f"read {len(rows)} labelled posts; filled: " + ", ".join(f"{k} {v}" for k, v in cov.items()))
    blank_core = [r["gold_id"] for r in rows if any(r[k] is None for k in ("subtopic_id", "content_type", "difficulty"))]
    if blank_core:
        print(f"note: main labels still blank for {', '.join(blank_core)} (they are skipped in the comparison)")
    print(f"clean labels : {labels}")
    print(f"post ids     : {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
