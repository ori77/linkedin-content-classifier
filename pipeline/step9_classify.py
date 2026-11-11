"""Step 9: classify saved posts on all 7 lenses with a local Ollama model.

In simple terms: for every post in the CSV this script asks the model two small questions
(which subtopic? / everything else), does the arithmetic with the rubrics, and appends one line
per post to a .jsonl file. It remembers what it already did (by a fingerprint of the post,
the prompts, the model, the settings, the taxonomy and the rubrics), so you can stop it at any
time and run the same command again to continue. Change anything that matters and only the
affected posts are redone.

Examples (run from the repo root, after `ollama pull qwen3:8b`):

  # look at the prompts for the first post, without calling the model
  python pipeline/step9_classify.py --csv posts_with_save_dates.csv --taxonomy taxonomy/taxonomy.v1.yaml ^
      --assignments taxonomy/taxonomy_assignments.v1.csv --show-prompt

  # classify 5 posts to try it out
  python pipeline/step9_classify.py --csv ... --taxonomy ... --assignments ... --limit 5

  # classify everything, then re-run the same command to resume after a stop
  python pipeline/step9_classify.py --csv ... --taxonomy ... --assignments ...

Exit status: 0 = all done, 1 = some posts failed (listed in <out>.failures.jsonl), 2 = cannot start
(bad input file, Ollama not running).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import classifier as C
import classify_inputs as I
import rubrics as R
import scoring
from ollama_json import OLLAMA_URL, OllamaUnavailable, chat_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_out(model: str) -> str:
    return f"classifications_{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}.jsonl"


def read_done_keys(path: Path) -> set[str]:
    """Cache keys already stored. A half-written last line (after a crash) is ignored."""
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["cache_key"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return keys


def append_line(path: Path, record: dict) -> None:
    """Append one JSON line and force it to disk, so a crash loses at most the current post."""
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as fh:
        if needs_newline:                      # a crash may have left a half-written last line
            fh.write("\n")
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify saved posts on all 7 lenses (Step 9).")
    p.add_argument("--csv", required=True, help="posts file, e.g. posts_with_save_dates.csv")
    p.add_argument("--taxonomy", required=True, help="taxonomy/taxonomy.v1.yaml")
    p.add_argument("--assignments", required=True, help="taxonomy/taxonomy_assignments.v1.csv")
    p.add_argument("--rubrics", default=str(R.DEFAULT_PATH), help="rubrics/rubrics.v1.yaml")
    p.add_argument("--model", default="qwen3:latest")
    p.add_argument("--out", help="output .jsonl (default: classifications_<model>.jsonl)")
    p.add_argument("--limit", type=int, help="classify at most this many NEW posts, then stop")
    p.add_argument("--ids", help="comma-separated activity ids to classify (others are ignored)")
    p.add_argument("--url", default=OLLAMA_URL)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--think", action="store_true", help="let the model think first (slower)")
    p.add_argument("--presence-penalty", type=float, help="override Ollama's default for this model")
    p.add_argument("--show-prompt", action="store_true", help="print the prompts for the first post and stop")
    return p


def main(argv: list[str] | None = None, chat=chat_json) -> int:
    args = build_parser().parse_args(argv)
    try:
        rub = R.load_rubrics(args.rubrics)
        tax = I.load_taxonomy(args.taxonomy)
        assignments = I.load_assignments(args.assignments, tax)
        posts, skipped = I.load_posts(args.csv)
    except (R.RubricsError, I.InputError) as exc:
        print(exc)
        return 2

    if args.ids:
        wanted = set()
        for part in args.ids.split(","):
            try:
                wanted.add(int(part.strip()))
            except ValueError:
                print(f"--ids: '{part.strip()}' is not a whole number")
                return 2
        missing = wanted - {p.activity_id for p in posts}
        if missing:
            print(f"--ids: not found (or without text): {sorted(missing)}")
            return 2
        posts = [p for p in posts if p.activity_id in wanted]

    options = {"presence_penalty": args.presence_penalty} if args.presence_penalty is not None else None
    ctx = C.Context(model=args.model, taxonomy=tax, rubrics=rub, assignments=assignments, url=args.url,
                    num_ctx=args.num_ctx, think=args.think, options=options)

    if args.show_prompt:
        if not posts:
            print("No posts to show.")
            return 2
        post = posts[0]
        baseline = scoring.baseline_cover_minutes(post.word_count, rub)
        user_a, user_b = C.user_prompt_a(post), C.user_prompt_b(post, baseline)
        for title, text in (("SYSTEM PROMPT, CALL A (subtopic)", ctx.system_a), ("USER PROMPT, CALL A", user_a),
                            ("SYSTEM PROMPT, CALL B (everything else)", ctx.system_b), ("USER PROMPT, CALL B", user_b)):
            print(f"===== {title} (about {C.estimate_tokens(text)} tokens) =====\n{text}\n")
        print(f"Largest prompt is about {max(C.estimate_tokens(ctx.system_a + user_a), C.estimate_tokens(ctx.system_b + user_b))}"
              f" tokens of the {args.num_ctx} available (answers need room too).")
        return 0

    out = Path(args.out or default_out(args.model))
    failures_path = out.with_name(out.name + ".failures.jsonl")
    done = read_done_keys(out)
    todo = [p for p in posts if C.cache_key(p, ctx) not in done]
    already = len(posts) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(posts)} posts with text ({len(skipped)} without text skipped); {already} already done; "
          f"{len(todo)} to do with {args.model} -> {out}")

    stored: list[dict] = []
    failed = 0
    try:
        for n, post in enumerate(todo, start=1):
            try:
                rec = C.classify_post(post, ctx, chat=chat)
            except C.ClassifyFailed as exc:
                failed += 1
                append_line(failures_path, {"activity_id": post.activity_id, "model": args.model,
                                            "error": str(exc), "created_at": utc_now()})
                print(f"[{n}/{len(todo)}] {post.activity_id}  FAILED: {exc}")
                continue
            except OllamaUnavailable as exc:
                print(f"Stopping: {exc}\nRun the same command again to continue.")
                return 2
            rec["created_at"] = utc_now()
            append_line(out, rec)
            stored.append(rec)
            avg = sum(r["seconds"] for r in stored) / len(stored)
            eta_min = avg * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)}] {rec['activity_id']}  {rec['subtopic_id']}  {rec['content_type']}/{rec['difficulty']}"
                  f"  align {rec['alignment_score']}  conf {rec['confidence']}{'  REVIEW' if rec['needs_review'] else ''}"
                  f"  {rec['seconds']}s  (about {eta_min:.0f} min left)")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
        return 130

    basis = {b: sum(1 for r in stored if r["time"]["basis"] == b) for b in R.BASIS_IDS}
    print(f"\nDone: {len(stored)} classified, {already} were already done, {failed} failed.")
    if stored:
        print(f"Needs review: {sum(r['needs_review'] for r in stored)}. Time basis: "
              + ", ".join(f"{k} {v}" for k, v in basis.items())
              + f". Average {sum(r['seconds'] for r in stored) / len(stored):.1f} s per post.")
    if failed:
        print(f"Failures are listed in {failures_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
