"""Mutation check for pipeline/rubrics.py (Step 8).

In simple terms: we deliberately break the validator, one small change at a time, in a
throwaway copy of the project. After each break we run the tests. A good test suite FAILS
(the break is "killed"). If the tests still pass (the break "survived"), a rule is not
really being tested and we need another test.

Run:  python tools/mutation_check_rubrics.py        (exit 0 = all killed, 1 = some survived)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("pipeline") / "rubrics.py"

# (name, text that must appear exactly once, replacement)
MUTATIONS = [
    ("sha: stop normalising CRLF", 'raw.replace(b"\\r\\n", b"\\n").replace(b"\\r", b"\\n")', "raw"),
    ("ids: ignore missing ids", "    if missing:\n", "    if False:\n"),
    ("ids: ignore duplicate ids", "    if len(found) != len(set(found)):", "    if False:"),
    ("ids: ignore order", "if ok and ordered and found != expected", "if ok and False and found != expected"),
    ("content_type: allow catch-all anywhere", "ids[-1] != CATCH_ALL", "False"),
    ("alignment: weights tolerance huge", "abs(total - 1.0) > 1e-6", "abs(total - 1.0) > 1.0"),
    ("alignment: allow zero weight", "not _is_num(w) or w <= 0", "not _is_num(w) or w < 0"),
    ("number check: accept True as a number", "and not isinstance(x, bool)", ""),
    ("bands: allow equal mins", "any(a <= b for a, b in zip(mins, mins[1:]))", "any(a < b for a, b in zip(mins, mins[1:]))"),
    ("bands: last band need not start at 0", "mins[-1] != 0", "False"),
    ("time: technical may be faster than reading", "elif _is_num(r) and tech > r:", "elif _is_num(r) and tech > r * 100:"),
    ("time: allow zero reading speed", 'if not _is_num(r) or r <= 0:', 'if not _is_num(r) or r < 0:'),
    ("hands_on: allow negative floors", "not _is_num(v) or v < 0", "not _is_num(v) or v < -100"),
    ("hands_on: allow empty applies list", "not isinstance(applies, list) or not applies or", "not isinstance(applies, list) or"),
    ("cover_clamp: drop 'low <= 1 <= high'", 'not 0 < c["low"] <= 1 <= c["high"]', 'not 0 < c["low"] <= c["high"]'),
    ("limits: drop cover_max <= understand_max", '0 < lim["cover_max"] <= lim["understand_max"]', '0 < lim["cover_max"]'),
    ("depth: allow decreasing multipliers", "any(a > b for a, b in zip(vals, vals[1:]))", "any(a > b + 100 for a, b in zip(vals, vals[1:]))"),
    ("combine: accept any method", 'u.get("combine") != "geometric_mean"', "False"),
    ("clamp_vs_cover: allow low below 1", 'cl["low"] < 1', 'cl["low"] < 0'),
    ("clamp_vs_cover: allow high <= low", 'cl["high"] <= cl["low"]', 'cl["high"] < 0'),
    ("news cap: allow below 1", "cap < 1", "cap < 0"),
    ("round_nice: last entry may have 'below'", 'if last and "below" in step', 'if False and "below" in step'),
    ("round_nice: allow non-increasing 'below'", "any(a >= b for a, b in zip(belows, belows[1:]))", "any(a > b + 1000 for a, b in zip(belows, belows[1:]))"),
    ("buckets: wrong label count accepted", "len(labels) != len(edges) + 1", "len(labels) != len(edges)"),
    ("buckets: allow repeated edges", "any(a >= c for a, c in zip(edges, edges[1:]))", "any(a > c for a, c in zip(edges, edges[1:]))"),
    ("summary: target may exceed hard limit", 'tw["max"] <= hw["max"]', "True"),
    ("summary: single_paragraph not checked", 'isinstance(s.get("single_paragraph"), bool)', "True"),
    ("load: never raise on problems", "    if problems:\n        raise RubricsError", "    if False:\n        raise RubricsError"),
    ("cli: always exit 0", "            print(f\"  - {p}\")\n        return 1", "            print(f\"  - {p}\")\n        return 0"),
]


def run_suite(workdir: Path) -> bool:
    """True if the suite passes in workdir."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_rubrics.py", "-x", "-q",
                        "-p", "no:cacheprovider"], cwd=workdir, capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    original = (ROOT / TARGET).read_text(encoding="utf-8")
    survivors, unusable = [], []
    for name, old, new in MUTATIONS:
        if original.count(old) != 1:
            unusable.append((name, original.count(old)))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "proj"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".pytest_cache", "tools"))
            (work / TARGET).write_text(original.replace(old, new), encoding="utf-8")
            killed = not run_suite(work)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {name}")
        if not killed:
            survivors.append(name)
    for name, n in unusable:
        print(f"UNUSABLE  {name} (pattern found {n} times, needs exactly 1)")
    total = len(MUTATIONS)
    print(f"\n{total - len(survivors) - len(unusable)} of {total} killed, "
          f"{len(survivors)} survived, {len(unusable)} unusable")
    return 1 if survivors or unusable else 0


if __name__ == "__main__":
    sys.exit(main())
