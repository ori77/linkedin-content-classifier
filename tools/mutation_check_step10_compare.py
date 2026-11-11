"""Mutation check for pipeline/step10_compare.py (Step 10c). Break one line at a time; tests must notice.
Run:  python tools/mutation_check_step10_compare.py        (exit 0 = all killed, 1 = some survived)

In simple terms: we deliberately break the comparison script in a throwaway copy of the project, one small
change at a time, and run the tests. A good test suite FAILS after each break (the break is "killed").
If the tests still pass, that line is not really tested and we need another test.

Not listed on purpose (no test could tell the difference): rounding the total seconds (round(x, 1) of a
number that is already exact), and the order of the two output files in the returned list beyond what
test_write_outputs already pins.
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    # helpers
    ("topic_of: whole id returned", 'return subtopic_id.split(".", 1)[0]', "return subtopic_id"),
    ("topic_of: split at every dot", 'subtopic_id.split(".", 1)[0]', 'subtopic_id.split(".")[-1]'),
    ("_same: blank label counted as wrong", "return None if label is None else label == pred", "return label == pred"),
    ("_blank_to_none: spaces kept", 'value = (value or "").strip()', 'value = (value or "")'),
    ("_read_csv: missing columns not noticed", "missing = [c for c in needed if c not in (reader.fieldnames or [])]", "missing = []"),
    # reading labels
    ("labels: reviewed file not recognised", 'reviewed = "original_subtopic_id" in header', "reviewed = False"),
    ("labels: original set reads the reviewed columns", '{"original": "original_", "reviewed": ""}', '{"original": "", "reviewed": ""}'),
    ("labels: duplicate ids accepted", "if aid in out:", "if False:"),
    ("labels: empty file accepted", "if not rows:", "if False:"),
    ("labels: ids not stripped", 'aid = row["activity_id"].strip()', 'aid = row["activity_id"]'),
    # reading the key and the run
    ("key: None not handled", "    if path is None:\n        return {}", "    if False:\n        return {}"),
    ("run: blank lines not skipped", "                if not line.strip():\n                    continue", "                if False:\n                    continue"),
    ("run: first record wins", 'out[str(rec["activity_id"])] = rec', 'out.setdefault(str(rec["activity_id"]), rec)'),
    ("run: any dict accepted", 'or "activity_id" not in rec or "subtopic_id" not in rec:', ":"),
    ("run: not a dict accepted", "if not isinstance(rec, dict) or", "if False or"),
    ("run: empty run accepted", "    if not out:\n        raise CompareError(f\"run", "    if False:\n        raise CompareError(f\"run"),
    # auc
    ("auc: ties count as wins", "1.0 if p > n else 0.5 if p == n else 0.0", "1.0 if p >= n else 0.0"),
    ("auc: ties count as losses", "1.0 if p > n else 0.5 if p == n else 0.0", "1.0 if p > n else 0.0"),
    ("auc: wrong direction", "1.0 if p > n else 0.5 if p == n else 0.0", "1.0 if p < n else 0.5 if p == n else 0.0"),
    ("auc: one empty group not handled", "if not pos or not neg:", "if not pos and not neg:"),
    ("auc: wrong divisor", "wins / (len(pos) * len(neg))", "wins / len(pos)"),
    # per-post rows
    ("rows: difficulty delta reversed", 'DIFFICULTY_ORDER.index(pred_diff) - DIFFICULTY_ORDER.index(lab["difficulty"])',
     'DIFFICULTY_ORDER.index(lab["difficulty"]) - DIFFICULTY_ORDER.index(pred_diff)'),
    ("rows: unknown predicted level crashes", 'if lab["difficulty"] in DIFFICULTY_ORDER and pred_diff in DIFFICULTY_ORDER:', 'if lab["difficulty"] in DIFFICULTY_ORDER:'),
    ("rows: blank gold subtopic gives a topic", 'label_topic = topic_of(lab["subtopic_id"]) if lab["subtopic_id"] else None', 'label_topic = topic_of(lab["subtopic_id"] or "")'),
    ("rows: clustering compared without a key", '"clustering_subtopic_ok": None if clustering is None else _same(lab["subtopic_id"], clustering),', '"clustering_subtopic_ok": _same(lab["subtopic_id"], clustering),'),
    ("rows: clustering topic compared without a key", '"clustering_topic_ok": None if clustering is None else _same(label_topic, topic_of(clustering)),', '"clustering_topic_ok": _same(label_topic, topic_of(clustering)),'),
    ("rows: blank clustering cell counted", 'key.get(aid, {}).get("clustering_subtopic") or None', 'key.get(aid, {}).get("clustering_subtopic")'),
    ("rows: needs_review not turned into a bool", 'bool(rec.get("needs_review"))', 'rec.get("needs_review")'),
    ("rows: missing posts not skipped", "        if rec is None:\n            missing.append", "        if False:\n            missing.append"),
    ("rows: topic always counted right", '"topic_ok": _same(label_topic, topic_of(rec["subtopic_id"])),', '"topic_ok": _same(label_topic, label_topic),'),
    # tallies and summary
    ("tally: wrong counted as right", 'sum(1 for f in flags if f)', "sum(1 for f in flags)"),
    ("tally: blank counted in the total", "flags = [r[field] for r in rows if r[field] is not None]", "flags = [r[field] for r in rows]"),
    ("summary: blank types enter the confusion table", '    for r in rows:\n        if r["label_content_type"]:', '    for r in rows:\n        if True:'),
    ("summary: confidence not checked for numbers", ' and isinstance(r["confidence"], (int, float))]', "]"),
    ("summary: retries in the topic call ignored", '(r["attempts_topic"] or 1) > 1 or ', ""),
    ("summary: retries in the other call ignored", ' or (r["attempts_rest"] or 1) > 1', ""),
    ("summary: group without a name keeps an empty name", 'strata[name or "(none)"]', "strata[name]"),
    ("summary: needs_review not counted", 'sum(1 for r in rows if r["needs_review"])', "0"),
    ("summary: mean seconds wrong", 'round(sum(seconds) / len(seconds), 1)', "round(sum(seconds), 1)"),
    # showing and saving
    ("format: baseline row always shown", 'if first["clustering"]["subtopic"]["total"]:', "if True:"),
    ("format: empty tally not shown as a dash", 'return f"{t[\'right\']}/{t[\'total\']}" if t["total"] else "-"', 'return f"{t[\'right\']}/{t[\'total\']}"'),
    ("format: missing posts not listed", "            if s[\"missing_from_run\"]:", "            if False:"),
    ("write: output folder not created", "out_dir.mkdir(parents=True, exist_ok=True)", "pass"),
    ("write: csv without a BOM", 'open(pp, "w", newline="", encoding="utf-8-sig")', 'open(pp, "w", newline="", encoding="utf-8")'),
    ("write: json shortened", "json.dumps(report, indent=2, ensure_ascii=False)", "json.dumps(report)[:5]"),
    ("run_name: prefix kept", 'stem[len("classifications_"):] if stem.startswith("classifications_") else stem', "stem"),
    # command line
    ("cli: duplicate run names accepted", "            if name in runs:", "            if False:"),
    ("cli: key without every gold post accepted", "if key and gold_ids - set(key):", "if False:"),
    ("cli: a run with none of the gold posts accepted", "            if not gold_ids & set(run):", "            if False:"),
    ("cli: problems still exit 0", '        print(f"PROBLEMS:\\n- {exc}")\n        return 1', '        print(f"PROBLEMS:\\n- {exc}")\n        return 0'),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/step10_compare.py", "tests/test_step10_compare.py", MUTATIONS))
