"""Mutation check for pipeline/step10_gold_labels.py (Step 10b).
Run:  python tools/mutation_check_step10_labels.py        (exit 0 = all killed)
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    ("blank: whitespace counts as a value", 'isinstance(v, str) and v.strip() == ""', 'isinstance(v, str) and v == ""'),
    ("blank: None counts as a value", "return v is None or (isinstance", "return (isinstance"),
    ("number: booleans accepted", "if isinstance(v, bool):\n        return None", "if False:\n        return None"),
    ("number: text never read as a number", "    if isinstance(v, str):\n        try:", "    if False:\n        try:"),
    ("number: non-finite accepted", "and math.isfinite(v):", ":"),
    ("subtopic: separator ignored", "str(v).split(SUBTOPIC_SEP)[0].strip()", "str(v).strip()"),
    ("subtopic: unknown id accepted", "if sid in tax.choice_ids:", "if True:"),
    ("subtopic: other.general rejected", "if sid in tax.choice_ids:", "if sid in [s.id for s in tax.subtopics]:"),
    ("type: unknown accepted", "if str(v).strip() in valid:", "if True:"),
    ("type: whitespace not trimmed", "out[key] = str(v).strip()", "out[key] = str(v)"),
    ("alignment: upper limit missing", "if n is None or not 0 <= n <= 10:", "if n is None or not 0 <= n:"),
    ("alignment: lower limit missing", "if n is None or not 0 <= n <= 10:", "if n is None or not n <= 10:"),
    ("alignment: zero rejected", "if n is None or not 0 <= n <= 10:", "if n is None or not 0 < n <= 10:"),
    ("alignment: ten rejected", "if n is None or not 0 <= n <= 10:", "if n is None or not 0 <= n < 10:"),
    ("minutes: zero accepted", "if n is None or n < 1 or n != int(n):", "if n is None or n < 0 or n != int(n):"),
    ("minutes: one rejected", "if n is None or n < 1 or n != int(n):", "if n is None or n < 2 or n != int(n):"),
    ("minutes: fractions accepted", "if n is None or n < 1 or n != int(n):", "if n is None or n < 1:"),
    ("minutes: kept as float", "out[key] = int(n)", "out[key] = n"),
    ("notes: spacing kept", 'out["notes"] = " ".join(str(row["notes"]).split())', 'out["notes"] = str(row["notes"])'),
    ("read: warnings not silenced", 'warnings.simplefilter("ignore")', "pass"),
    ("read: unreadable file not caught", "except (OSError, ValueError, KeyError, zipfile.BadZipFile, InvalidFileException) as exc:",
     "except (OSError, ValueError, KeyError) as exc:"),
    ("read: sheet name unchecked", "if G.SHEET_NAME not in wb.sheetnames:", "if False:"),
    ("read: empty sheet accepted", "if not rows:", "if False:"),
    ("read: missing columns accepted", "if missing:\n        raise LabelError", "if False:\n        raise LabelError"),
    ("read: blank lines read as posts", "if all(_blank(v) for v in values):", "if False:"),
    ("read: empty gold id accepted", "if not gid:\n            problems.append", "if False:\n            problems.append"),
    ("read: duplicate gold id accepted", "elif gid in seen:", "elif False:"),
    ("read: activity id may be a number", "if isinstance(aid_raw, str) and aid_raw.strip().isdigit():", "if str(aid_raw).strip().isdigit():"),
    ("read: key id mismatch accepted", "elif key_ids[gid] != aid:", "elif False:"),
    ("read: unknown gold id accepted", "if gid not in key_ids:", "if False:"),
    ("read: row missing from sheet accepted", "for gid in sorted(set(key_ids) - seen):", "for gid in []:"),
    ("read: problems do not stop", "    if problems:\n        raise LabelError(\"labels problems", "    if False:\n        raise LabelError(\"labels problems"),
    ("key: BOM not skipped", 'Path(path).open(newline="", encoding="utf-8-sig")', 'Path(path).open(newline="", encoding="utf-8")'),
    ("coverage: counts blanks", "sum(1 for r in rows if r[k] is not None)", "sum(1 for r in rows)"),
    ("coverage: understand minutes missing", ', "cover_minutes", "understand_minutes")}', ', "cover_minutes")}'),
    ("output: ids separator", '",".join(str(r["activity_id"]) for r in rows)', '" ".join(str(r["activity_id"]) for r in rows)'),
    ("main: errors give exit 0", "        return 1\n    cov", "        return 0\n    cov"),
    ("main: key not used", "key = read_key(args.key) if args.key else None", "key = None"),
    ("main: blank note missing", "if blank_core:", "if False:"),
    ("main: blank check ignores difficulty", 'for k in ("subtopic_id", "content_type", "difficulty"))]', 'for k in ("subtopic_id", "content_type"))]'),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/step10_gold_labels.py", "tests/test_step10_gold_labels.py", MUTATIONS))
