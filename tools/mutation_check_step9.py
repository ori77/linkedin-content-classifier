"""Mutation check for pipeline/step9_classify.py (Step 9). Slower (each run starts a fake server).
Run:  python tools/mutation_check_step9.py        (exit 0 = all killed)

Not listed on purpose: `os.fsync` (a test cannot see whether bytes reached the disk) and the
Ctrl+C handler (needs a real keyboard interrupt).
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    ("default_out: dots turned into dashes", "[^A-Za-z0-9._-]+", "[^A-Za-z0-9]+"),
    ("read_done_keys: stray lines crash it", "except (json.JSONDecodeError, KeyError, TypeError):", "except json.JSONDecodeError:"),
    ("append_line: torn last line not separated", "needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b\"\\n\")", "needs_newline = False"),
    ("append_line: non-ASCII escaped", "ensure_ascii=False", "ensure_ascii=True"),
    ("main: input errors not caught", "except (R.RubricsError, I.InputError) as exc:", "except R.RubricsError as exc:"),
    ("main: unknown ids ignored", "        if missing:\n", "        if False:\n"),
    ("main: --limit ignored", "if args.limit is not None:", "if False:"),
    ("main: finished posts redone", "todo = [p for p in posts if C.cache_key(p, ctx) not in done]", "todo = list(posts)"),
    ("main: presence penalty not sent", "options = {\"presence_penalty\": args.presence_penalty} if args.presence_penalty is not None else None", "options = None"),
    ("main: --think ignored", "num_ctx=args.num_ctx, think=args.think,", "num_ctx=args.num_ctx, think=False,"),
    ("main: --num-ctx ignored", "num_ctx=args.num_ctx, think=args.think,", "num_ctx=8192, think=args.think,"),
    ("main: --url ignored", "assignments=assignments, url=args.url,", "assignments=assignments,"),
    ("main: failures not counted", "                failed += 1\n", "                pass\n"),
    ("main: failures not written", "                append_line(failures_path,", "                (lambda *a: None)(failures_path,"),
    ("main: failed run exits 0", "return 1 if failed else 0", "return 0"),
    ("main: Ollama missing exits 0", "Run the same command again to continue.\")\n                return 2", "Run the same command again to continue.\")\n                return 0"),
    ("main: created_at not stored", '            rec["created_at"] = utc_now()\n', ""),
    ("main: results not written", "            append_line(out, rec)\n", ""),
    ("main: --show-prompt calls the model", "    if args.show_prompt:\n", "    if False:\n"),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/step9_classify.py", "tests/test_step9_classify.py", MUTATIONS))
