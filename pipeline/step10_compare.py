"""Step 10c: score one or more classifier runs against the hand-made gold labels.

In simple terms
---------------
You labelled 30 posts by hand (gold labels). The classifier has also labelled those posts, and
clustering gave each post a subtopic earlier. This script lines the three up, post by post, and
counts how often each one matches YOUR label:

  * subtopic     the exact subtopic id is the same
  * topic        the part before the dot is the same (a softer test)
  * content type tutorial / news / interview_question_set ... is the same
  * difficulty   beginner / intermediate / advanced / expert is the same

It also reports (so a model can be judged for more than accuracy): how the classifier's confidence
relates to being right, how often it had to re-ask the model, how long it took, and where it
disagrees (which content types, whether it is too easy or too hard on difficulty).

Several runs can be given at once (one classifications_*.jsonl file per model), and Step 11 uses it
for every model. A blank gold label is skipped for that lens (a blank means "no opinion").
When the labels file is gold_labels_reviewed.csv, TWO sets are scored: "original" (your first
labels) and "reviewed" (after you accepted some of Claude's suggestions), so you can see how much
a score depends on those corrections.

Run:
    python pipeline/step10_compare.py --labels gold/gold_labels_reviewed.csv --key gold/gold_key.csv \\
        --runs out/classifications_qwen3-latest.jsonl out/classifications_gemma3-12b.jsonl
Writes gold/compare_report.json (all numbers) and gold/compare_posts.csv (one row per post and run).
Exit status 0 = fine, 1 = an input file has a problem (the problem is printed).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced", "expert"]    # easiest first
LENSES = ("subtopic", "topic", "content_type", "difficulty")
REPORT_FILE = "compare_report.json"
POSTS_FILE = "compare_posts.csv"
REVIEWED_COLUMNS = ["gold_id", "activity_id", "original_subtopic_id", "subtopic_id", "original_content_type",
                    "content_type", "original_difficulty", "difficulty"]
PLAIN_COLUMNS = ["gold_id", "activity_id", "subtopic_id", "content_type", "difficulty"]
KEY_COLUMNS = ["gold_id", "activity_id", "stratum", "clustering_subtopic"]
POST_COLUMNS = ["label_set", "run", "gold_id", "stratum", "label_subtopic", "pred_subtopic", "clustering_subtopic",
                "subtopic_ok", "topic_ok", "clustering_subtopic_ok", "clustering_topic_ok",
                "label_content_type", "pred_content_type", "content_type_ok",
                "label_difficulty", "pred_difficulty", "difficulty_ok", "difficulty_delta",
                "confidence", "needs_review", "attempts_topic", "attempts_rest", "basis", "seconds"]


class CompareError(Exception):
    """An input file has a problem; the message says which file and what."""


# ------------------------------------------------------------------ small helpers

def topic_of(subtopic_id: str) -> str:
    """'ai-agent-development.ai-agent-development' -> 'ai-agent-development'; 'other.general' -> 'other'."""
    return subtopic_id.split(".", 1)[0]


def _read_csv(path: Path, needed: list[str], what: str) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in needed if c not in (reader.fieldnames or [])]
            if missing:
                raise CompareError(f"{what} {path.name}: missing columns {missing}")
            return list(reader)
    except OSError as exc:
        raise CompareError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise CompareError(f"{what} {path.name} is not valid UTF-8: {exc}") from exc


def _blank_to_none(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


# ------------------------------------------------------------------ reading the inputs

def read_label_sets(path: Path | str) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {set name: {activity_id: {gold_id, subtopic_id, content_type, difficulty}}}.

    gold_labels_reviewed.csv gives two sets, "original" and "reviewed"; a plain gold_labels.csv gives one, "labels".
    """
    path = Path(path)
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except OSError as exc:
        raise CompareError(f"cannot read labels {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise CompareError(f"labels {path.name} is not valid UTF-8: {exc}") from exc
    reviewed = "original_subtopic_id" in header
    rows = _read_csv(path, REVIEWED_COLUMNS if reviewed else PLAIN_COLUMNS, "labels")
    prefixes = {"original": "original_", "reviewed": ""} if reviewed else {"labels": ""}
    sets: dict[str, dict[str, dict[str, Any]]] = {}
    for name, prefix in prefixes.items():
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            aid = row["activity_id"].strip()
            if aid in out:
                raise CompareError(f"labels {path.name}: activity_id {aid} appears twice")
            out[aid] = {"gold_id": row["gold_id"].strip(),
                        "subtopic_id": _blank_to_none(row[prefix + "subtopic_id"]),
                        "content_type": _blank_to_none(row[prefix + "content_type"]),
                        "difficulty": _blank_to_none(row[prefix + "difficulty"])}
        sets[name] = out
    if not rows:
        raise CompareError(f"labels {path.name}: no rows")
    return sets


def read_key(path: Path | str | None) -> dict[str, dict[str, str]]:
    """{activity_id: {stratum, clustering_subtopic}} from gold_key.csv (empty if no key is given)."""
    if path is None:
        return {}
    rows = _read_csv(Path(path), KEY_COLUMNS, "key")
    return {r["activity_id"].strip(): {"stratum": r["stratum"].strip(),
                                       "clustering_subtopic": r["clustering_subtopic"].strip()} for r in rows}


def read_run(path: Path | str) -> dict[str, dict[str, Any]]:
    """{activity_id: classification record} from a classifications_*.jsonl file (the last record of a post wins)."""
    path = Path(path)
    out: dict[str, dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CompareError(f"run {path.name} line {n}: not valid JSON ({exc.msg})") from exc
                if not isinstance(rec, dict) or "activity_id" not in rec or "subtopic_id" not in rec:
                    raise CompareError(f"run {path.name} line {n}: not a classification record")
                out[str(rec["activity_id"])] = rec
    except OSError as exc:
        raise CompareError(f"cannot read run {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise CompareError(f"run {path.name} is not valid UTF-8: {exc}") from exc
    if not out:
        raise CompareError(f"run {path.name}: no records")
    return out


# ------------------------------------------------------------------ the numbers

def _same(label: str | None, pred: str | None) -> bool | None:
    """True/False, or None when the gold label is blank (that post does not count for this lens)."""
    return None if label is None else label == pred


def auc(scores: list[float], good: list[bool]) -> float | None:
    """Chance that a randomly picked correct post has a higher score than a wrong one (ties count half).

    1.0 = confidence separates right from wrong perfectly, 0.5 = no better than a coin, None = one group is empty.
    """
    pos = [s for s, g in zip(scores, good) if g]
    neg = [s for s, g in zip(scores, good) if not g]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def post_rows(labels: dict[str, dict[str, Any]], run: dict[str, dict[str, Any]],
              key: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per gold post that the run covers, plus the gold ids the run is missing."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for aid, lab in labels.items():
        rec = run.get(aid)
        if rec is None:
            missing.append(lab["gold_id"])
            continue
        clustering = key.get(aid, {}).get("clustering_subtopic") or None
        pred_diff = rec.get("difficulty")
        delta = None
        if lab["difficulty"] in DIFFICULTY_ORDER and pred_diff in DIFFICULTY_ORDER:
            delta = DIFFICULTY_ORDER.index(pred_diff) - DIFFICULTY_ORDER.index(lab["difficulty"])
        label_topic = topic_of(lab["subtopic_id"]) if lab["subtopic_id"] else None
        rows.append({
            "gold_id": lab["gold_id"], "stratum": key.get(aid, {}).get("stratum", ""),
            "label_subtopic": lab["subtopic_id"], "pred_subtopic": rec["subtopic_id"],
            "clustering_subtopic": clustering,
            "subtopic_ok": _same(lab["subtopic_id"], rec["subtopic_id"]),
            "topic_ok": _same(label_topic, topic_of(rec["subtopic_id"])),
            "clustering_subtopic_ok": None if clustering is None else _same(lab["subtopic_id"], clustering),
            "clustering_topic_ok": None if clustering is None else _same(label_topic, topic_of(clustering)),
            "label_content_type": lab["content_type"], "pred_content_type": rec.get("content_type"),
            "content_type_ok": _same(lab["content_type"], rec.get("content_type")),
            "label_difficulty": lab["difficulty"], "pred_difficulty": pred_diff,
            "difficulty_ok": _same(lab["difficulty"], pred_diff), "difficulty_delta": delta,
            "confidence": rec.get("confidence"), "needs_review": bool(rec.get("needs_review")),
            "attempts_topic": (rec.get("attempts") or {}).get("topic"),
            "attempts_rest": (rec.get("attempts") or {}).get("rest"),
            "basis": (rec.get("time") or {}).get("basis"), "seconds": rec.get("seconds"),
        })
    return rows, missing


def tally(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """{'right': n, 'total': n}: posts where the flag is True / posts where it is not None."""
    flags = [r[field] for r in rows if r[field] is not None]
    return {"right": sum(1 for f in flags if f), "total": len(flags)}


def _lens_tallies(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {"subtopic": tally(rows, "subtopic_ok"), "topic": tally(rows, "topic_ok"),
            "content_type": tally(rows, "content_type_ok"), "difficulty": tally(rows, "difficulty_ok")}


def summarise(rows: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    """All the numbers for one run against one label set."""
    conf_rows = [r for r in rows if r["subtopic_ok"] is not None and isinstance(r["confidence"], (int, float))]
    right = [r["confidence"] for r in conf_rows if r["subtopic_ok"]]
    wrong = [r["confidence"] for r in conf_rows if not r["subtopic_ok"]]
    confusion: dict[str, Counter] = {}
    for r in rows:
        if r["label_content_type"]:
            confusion.setdefault(r["label_content_type"], Counter())[r["pred_content_type"]] += 1
    deltas = Counter(r["difficulty_delta"] for r in rows if r["difficulty_delta"] is not None)
    strata: dict[str, dict[str, Any]] = {}
    for name in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == name]
        strata[name or "(none)"] = {"posts": len(part), **_lens_tallies(part)}
    seconds = [r["seconds"] for r in rows if isinstance(r["seconds"], (int, float))]
    return {
        "posts": len(rows), "missing_from_run": missing,
        **_lens_tallies(rows),
        "clustering": {"subtopic": tally(rows, "clustering_subtopic_ok"), "topic": tally(rows, "clustering_topic_ok")},
        "by_stratum": strata,
        "confidence": {"mean_when_right": sum(right) / len(right) if right else None,
                       "mean_when_wrong": sum(wrong) / len(wrong) if wrong else None,
                       "auc_subtopic": auc([r["confidence"] for r in conf_rows], [bool(r["subtopic_ok"]) for r in conf_rows])},
        "content_type_confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "difficulty_delta": {str(k): v for k, v in sorted(deltas.items())},   # predicted level minus your level
        "retried_posts": sum(1 for r in rows if (r["attempts_topic"] or 1) > 1 or (r["attempts_rest"] or 1) > 1),
        "needs_review": sum(1 for r in rows if r["needs_review"]),
        "basis": dict(Counter(r["basis"] for r in rows)),
        "seconds": {"total": round(sum(seconds), 1), "mean": round(sum(seconds) / len(seconds), 1) if seconds else None},
    }


def compare(label_sets: dict[str, dict[str, dict[str, Any]]], runs: dict[str, dict[str, dict[str, Any]]],
            key: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every run against every label set. Returns (report, per-post rows with set and run names)."""
    report: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for set_name, labels in label_sets.items():
        report[set_name] = {}
        for run_name, run in runs.items():
            rows, missing = post_rows(labels, run, key)
            report[set_name][run_name] = summarise(rows, missing)
            all_rows.extend({"label_set": set_name, "run": run_name, **r} for r in rows)
    return report, all_rows


# ------------------------------------------------------------------ showing and saving

def _frac(t: dict[str, int]) -> str:
    return f"{t['right']}/{t['total']}" if t["total"] else "-"


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for set_name, runs in report.items():
        first = next(iter(runs.values()))
        lines.append(f"\n=== label set: {set_name} ({first['posts']} posts) ===")
        lines.append(f"{'':28}{'subtopic':>10}{'topic':>8}{'type':>8}{'difficulty':>12}")
        if first["clustering"]["subtopic"]["total"]:
            c = first["clustering"]
            lines.append(f"{'clustering (baseline)':28}{_frac(c['subtopic']):>10}{_frac(c['topic']):>8}{'-':>8}{'-':>12}")
        for name, s in runs.items():
            lines.append(f"{name[:27]:28}{_frac(s['subtopic']):>10}{_frac(s['topic']):>8}"
                         f"{_frac(s['content_type']):>8}{_frac(s['difficulty']):>12}")
        for name, s in runs.items():
            conf = s["confidence"]
            fmt = lambda v: "-" if v is None else f"{v:.2f}"      # noqa: E731
            lines.append(f"\n{name}")
            if s["missing_from_run"]:
                lines.append(f"  NOT IN THIS RUN: {', '.join(s['missing_from_run'])}")
            lines.append(f"  confidence: mean {fmt(conf['mean_when_right'])} when the subtopic is right, "
                         f"{fmt(conf['mean_when_wrong'])} when wrong, AUC {fmt(conf['auc_subtopic'])}")
            lines.append(f"  re-asked posts {s['retried_posts']}/{s['posts']}, needs_review {s['needs_review']}, "
                         f"time basis {s['basis']}, seconds total {s['seconds']['total']} (mean {s['seconds']['mean']})")
            lines.append(f"  difficulty, predicted level minus yours: {s['difficulty_delta'] or '-'}")
            lines.append("  by group (posts; subtopic, topic, type, difficulty): " + "; ".join(
                f"{g}: {v['posts']}; {_frac(v['subtopic'])}, {_frac(v['topic'])}, {_frac(v['content_type'])}, {_frac(v['difficulty'])}"
                for g, v in s["by_stratum"].items()))
            for label, got in s["content_type_confusion"].items():
                lines.append(f"  type you chose {label}: classifier said {got}")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rp, pp = out_dir / REPORT_FILE, out_dir / POSTS_FILE
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(pp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=POST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return [rp, pp]


def run_name(path: Path) -> str:
    """'classifications_qwen3-latest.jsonl' -> 'qwen3-latest'."""
    stem = path.stem
    return stem[len("classifications_"):] if stem.startswith("classifications_") else stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score classifier runs against the gold labels.")
    ap.add_argument("--labels", required=True, help="gold/gold_labels_reviewed.csv or gold/gold_labels.csv")
    ap.add_argument("--key", help="gold/gold_key.csv (adds the clustering baseline and the group breakdown)")
    ap.add_argument("--runs", required=True, nargs="+", help="one or more classifications_*.jsonl files")
    ap.add_argument("--out-dir", default="gold")
    args = ap.parse_args(argv)
    try:
        label_sets = read_label_sets(args.labels)
        key = read_key(args.key)
        runs: dict[str, dict[str, dict[str, Any]]] = {}
        for p in map(Path, args.runs):
            name = run_name(p)
            if name in runs:
                raise CompareError(f"two runs are called '{name}': rename one file")
            runs[name] = read_run(p)
        gold_ids = set(next(iter(label_sets.values())))
        if key and gold_ids - set(key):
            raise CompareError(f"key {Path(args.key).name} does not list gold posts {sorted(gold_ids - set(key))[:5]}...")
        for name, run in runs.items():
            if not gold_ids & set(run):
                raise CompareError(f"run '{name}' has none of the gold posts (wrong file?)")
    except CompareError as exc:
        print(f"PROBLEMS:\n- {exc}")
        return 1
    report, rows = compare(label_sets, runs, key)
    print(format_report(report))
    for p in write_outputs(Path(args.out_dir), report, rows):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
