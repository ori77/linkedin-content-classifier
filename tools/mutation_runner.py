"""Shared engine for the mutation checks (used by mutation_check_*.py).

In simple terms: for each planned "break", copy the project to a temporary folder, apply the
break to one source file, run the tests, and report whether the tests noticed (killed) or not
(SURVIVED). Each break must match the source exactly once, otherwise it is reported UNUSABLE.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def suite_passes(workdir: Path, test_file: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", test_file, "-x", "-q", "-p", "no:cacheprovider"],
                       cwd=workdir, capture_output=True, text=True)
    return r.returncode == 0


def run_mutations(target: str, test_file: str, mutations: list[tuple[str, str, str]]) -> int:
    """mutations = [(name, text that must appear exactly once, replacement)]. Returns exit code."""
    original = (ROOT / target).read_text(encoding="utf-8")
    survivors, unusable = [], []
    for name, old, new in mutations:
        if original.count(old) != 1:
            unusable.append((name, original.count(old)))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "proj"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".pytest_cache", "tools"))
            (work / target).write_text(original.replace(old, new), encoding="utf-8")
            killed = not suite_passes(work, test_file)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {name}")
        if not killed:
            survivors.append(name)
    for name, n in unusable:
        print(f"UNUSABLE  {name} (pattern found {n} times, needs exactly 1)")
    total = len(mutations)
    print(f"\n{total - len(survivors) - len(unusable)} of {total} killed, "
          f"{len(survivors)} survived, {len(unusable)} unusable")
    return 1 if survivors or unusable else 0
