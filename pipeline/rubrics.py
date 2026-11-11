"""Load, validate and fingerprint the rubrics file (Step 8).

In simple terms: the rubrics file holds the rules the classifier follows (difficulty
levels, content-type rules, alignment weights, time-estimate numbers). This module
reads that file, checks it is consistent, and gives it a fingerprint (a hash). The
fingerprint goes into the cache key, so changing a rule means old answers are not reused.

Run from the command line to check a file:
    python rubrics.py                      (checks rubrics/rubrics.v1.yaml)
    python rubrics.py path/to/other.yaml
Exit status 0 = fine, 1 = problems (all problems are printed together).
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The ids below are fixed by the data model (LLD section 4: enums and AlignmentSubscores).
# The rubrics file may change wording and numbers, but not these names, because the
# classifier's JSON schema is built from them.
DIFFICULTY_IDS = ["beginner", "intermediate", "advanced", "expert"]  # easiest first
CONTENT_TYPE_IDS = ["interview_question_set", "full_project", "mini_project",
                    "tutorial", "news", "miscellaneous"]
ALIGNMENT_IDS = ["applicability", "technical_depth", "hands_on", "currency",
                 "career_value", "production_relevance"]
BASIS_IDS = ["stated_in_post", "estimated_from_post", "post_only"]
STATED_UNIT_IDS = ["minutes", "hours", "weeks", "questions", "other", "none"]
UNCONVERTIBLE_UNIT_IDS = ["other", "none"]       # no minutes_per_unit allowed for these
CATCH_ALL = "miscellaneous"          # must be the last content-type rule

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "rubrics" / "rubrics.v1.yaml"


class RubricsError(Exception):
    """Raised by load_rubrics when the file has problems; message lists all of them."""


@dataclass(frozen=True)
class Rubrics:
    data: dict[str, Any]
    sha: str          # fingerprint of the file content
    path: Path

    @property
    def version(self) -> str:
        return self.data["version"]


# ---------------------------------------------------------------- small helpers

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def rubrics_sha(raw: bytes) -> str:
    """Fingerprint of the file. Windows (CRLF) and Linux (LF) copies must give the same hash."""
    normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalised).hexdigest()


def _same_ids(found: list[str], expected: list[str], where: str, problems: list[str],
              ordered: bool = False) -> bool:
    """Check a list of ids is exactly the expected set (no duplicates, none missing or extra)."""
    ok = True
    if len(found) != len(set(found)):
        problems.append(f"{where}: duplicate ids {sorted({i for i in found if found.count(i) > 1})}")
        ok = False
    missing = [i for i in expected if i not in found]
    extra = [i for i in found if i not in expected]
    if missing:
        problems.append(f"{where}: missing ids {missing}")
        ok = False
    if extra:
        problems.append(f"{where}: unknown ids {extra}")
        ok = False
    if ok and ordered and found != expected:
        problems.append(f"{where}: ids must be in the order {expected}")
        ok = False
    return ok


def _section(data: dict, key: str, kind: type, problems: list[str]):
    value = data.get(key)
    if not isinstance(value, kind):
        problems.append(f"missing or wrong type: '{key}' (expected {kind.__name__})")
        return None
    return value


def _id_list(items: list, where: str, problems: list[str]) -> list[str]:
    ids = []
    for n, item in enumerate(items):
        if not isinstance(item, dict) or not _is_text(item.get("id")):
            problems.append(f"{where}[{n}]: needs an 'id'")
        else:
            ids.append(item["id"])
    return ids


# ---------------------------------------------------------------- the checks

def validate_rubrics(data: Any) -> list[str]:
    """Return a list of problems (empty list = the rubrics are consistent)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["the file must contain a mapping at the top level"]

    for key in ("version", "role_profile", "judging_basis"):
        if not _is_text(data.get(key)):
            problems.append(f"'{key}' must be a non-empty text")

    _check_difficulty(data, problems)
    _check_content_type(data, problems)
    _check_alignment(data, problems)
    _check_bands(data, problems)
    _check_time(data, problems)
    _check_summary(data, problems)
    return problems


def _check_difficulty(data: dict, problems: list[str]) -> None:
    items = _section(data, "difficulty", list, problems)
    if items is None:
        return
    ids = _id_list(items, "difficulty", problems)
    _same_ids(ids, DIFFICULTY_IDS, "difficulty", problems, ordered=True)
    for item in items:
        if isinstance(item, dict):
            for field in ("definition", "example"):
                if not _is_text(item.get(field)):
                    problems.append(f"difficulty '{item.get('id')}': '{field}' is empty")


def _check_content_type(data: dict, problems: list[str]) -> None:
    items = _section(data, "content_type", list, problems)
    if items is None:
        return
    ids = _id_list(items, "content_type", problems)
    if _same_ids(ids, CONTENT_TYPE_IDS, "content_type", problems) and ids[-1] != CATCH_ALL:
        # the first matching rule wins, so a catch-all anywhere else would hide the rules after it
        problems.append(f"content_type: '{CATCH_ALL}' must be the last rule (first match wins)")
    for item in items:
        if isinstance(item, dict) and not _is_text(item.get("rule")):
            problems.append(f"content_type '{item.get('id')}': 'rule' is empty")


def _check_alignment(data: dict, problems: list[str]) -> None:
    items = _section(data, "alignment", list, problems)
    if items is None:
        return
    ids = _id_list(items, "alignment", problems)
    _same_ids(ids, ALIGNMENT_IDS, "alignment", problems)
    total = 0.0
    weights_ok = True
    for item in items:
        if not isinstance(item, dict):
            continue
        w = item.get("weight")
        if not _is_num(w) or w <= 0:
            problems.append(f"alignment '{item.get('id')}': weight must be a number above 0")
            weights_ok = False
        else:
            total += w
        for field in ("question", "high", "low"):
            if not _is_text(item.get(field)):
                problems.append(f"alignment '{item.get('id')}': '{field}' is empty")
    if weights_ok and abs(total - 1.0) > 1e-6:
        problems.append(f"alignment: weights must sum to 1, they sum to {total:.4f}")


def _check_bands(data: dict, problems: list[str]) -> None:
    bands = _section(data, "alignment_bands", list, problems)
    if bands is None:
        return
    if not bands:
        problems.append("alignment_bands: needs at least one band")
        return
    mins = []
    for n, band in enumerate(bands):
        if not isinstance(band, dict) or not _is_num(band.get("min")) or not 0 <= band["min"] <= 10:
            problems.append(f"alignment_bands[{n}]: 'min' must be a number from 0 to 10")
            continue
        if not _is_text(band.get("label")):
            problems.append(f"alignment_bands[{n}]: 'label' is empty")
        mins.append(band["min"])
    if len(mins) == len(bands):
        if any(a <= b for a, b in zip(mins, mins[1:])):
            problems.append("alignment_bands: 'min' values must strictly decrease (highest band first)")
        if mins[-1] != 0:
            problems.append("alignment_bands: the last band must start at 0 so every score has a band")


def _check_time(data: dict, problems: list[str]) -> None:
    t = _section(data, "time", dict, problems)
    if t is None:
        return
    r, tech = t.get("reading_wpm"), t.get("technical_wpm")
    if not _is_num(r) or r <= 0:
        problems.append("time.reading_wpm must be a number above 0")
    if not _is_num(tech) or tech <= 0:
        problems.append("time.technical_wpm must be a number above 0")
    elif _is_num(r) and tech > r:
        problems.append("time.technical_wpm must not be faster than reading_wpm")

    basis = t.get("basis")
    if not isinstance(basis, list):
        problems.append("time.basis must be a list")
    else:
        ids = _id_list(basis, "time.basis", problems)
        _same_ids(ids, BASIS_IDS, "time.basis", problems)
        for item in basis:
            if isinstance(item, dict) and not _is_text(item.get("rule")):
                problems.append(f"time.basis '{item.get('id')}': 'rule' is empty")

    _check_hands_on(t, problems)
    _check_clamps(t, problems)
    _check_understand(t, problems)
    _check_rounding_and_buckets(t, problems)


def _check_hands_on(t: dict, problems: list[str]) -> None:
    floors = t.get("hands_on_floor_minutes")
    if not isinstance(floors, dict):
        problems.append("time.hands_on_floor_minutes must be a mapping")
    else:
        _same_ids(list(floors), CONTENT_TYPE_IDS, "time.hands_on_floor_minutes", problems)
        for k, v in floors.items():
            if not _is_num(v) or v < 0:
                problems.append(f"time.hands_on_floor_minutes '{k}': must be a number, 0 or more")
    applies = t.get("hands_on_applies_when_basis")
    if not isinstance(applies, list) or not applies or any(b not in BASIS_IDS for b in applies):
        problems.append(f"time.hands_on_applies_when_basis must be a non-empty list of {BASIS_IDS}")
    _check_stated_units(t, problems)


def _check_stated_units(t: dict, problems: list[str]) -> None:
    units = t.get("stated_units")
    if not isinstance(units, list):
        problems.append("time.stated_units must be a list")
        return
    ids = _id_list(units, "time.stated_units", problems)
    _same_ids(ids, STATED_UNIT_IDS, "time.stated_units", problems)
    for item in units:
        if not isinstance(item, dict) or item.get("id") not in STATED_UNIT_IDS:
            continue
        uid, per = item["id"], item.get("minutes_per_unit")
        if not _is_text(item.get("describe")):
            problems.append(f"time.stated_units '{uid}': 'describe' is empty")
        if uid in UNCONVERTIBLE_UNIT_IDS:
            if per is not None:
                problems.append(f"time.stated_units '{uid}': must not have minutes_per_unit (it cannot be converted)")
        elif not _is_num(per) or per <= 0:
            problems.append(f"time.stated_units '{uid}': minutes_per_unit must be a number above 0")


def _check_clamps(t: dict, problems: list[str]) -> None:
    c = t.get("cover_clamp")
    if not isinstance(c, dict) or not _is_num(c.get("low")) or not _is_num(c.get("high")):
        problems.append("time.cover_clamp needs numbers 'low' and 'high'")
    elif not 0 < c["low"] <= 1 <= c["high"]:
        problems.append("time.cover_clamp must satisfy 0 < low <= 1 <= high")
    when = t.get("cover_clamp_applies_when_basis")
    if not isinstance(when, list) or not when or any(b not in BASIS_IDS for b in when):
        problems.append(f"time.cover_clamp_applies_when_basis must be a non-empty list of {BASIS_IDS}")
    lim = t.get("limits_minutes")
    if (not isinstance(lim, dict) or not _is_num(lim.get("cover_max"))
            or not _is_num(lim.get("understand_max"))):
        problems.append("time.limits_minutes needs numbers 'cover_max' and 'understand_max'")
    elif not 0 < lim["cover_max"] <= lim["understand_max"]:
        problems.append("time.limits_minutes must satisfy 0 < cover_max <= understand_max")


def _check_understand(t: dict, problems: list[str]) -> None:
    u = t.get("understand")
    if not isinstance(u, dict):
        problems.append("time.understand must be a mapping")
        return
    depth = u.get("depth_multiplier")
    if not isinstance(depth, dict):
        problems.append("time.understand.depth_multiplier must be a mapping")
    elif _same_ids(list(depth), DIFFICULTY_IDS, "time.understand.depth_multiplier", problems):
        vals = [depth[d] for d in DIFFICULTY_IDS]
        if any(not _is_num(v) or v <= 0 for v in vals):
            problems.append("time.understand.depth_multiplier: every value must be a number above 0")
        elif any(a > b for a, b in zip(vals, vals[1:])):
            problems.append("time.understand.depth_multiplier must not decrease with difficulty")
    typ = u.get("type_multiplier")
    if not isinstance(typ, dict):
        problems.append("time.understand.type_multiplier must be a mapping")
    elif _same_ids(list(typ), CONTENT_TYPE_IDS, "time.understand.type_multiplier", problems):
        if any(not _is_num(v) or v <= 0 for v in typ.values()):
            problems.append("time.understand.type_multiplier: every value must be a number above 0")
    if u.get("combine") != "geometric_mean":
        problems.append("time.understand.combine must be 'geometric_mean' (the only supported way)")
    cl = u.get("clamp_vs_cover")
    if not isinstance(cl, dict) or not _is_num(cl.get("low")) or not _is_num(cl.get("high")):
        problems.append("time.understand.clamp_vs_cover needs numbers 'low' and 'high'")
    else:
        # the database requires understand >= cover, so the lower clamp may not go below 1
        if cl["low"] < 1:
            problems.append("time.understand.clamp_vs_cover.low must be at least 1 (understand >= cover)")
        if cl["high"] <= cl["low"]:
            problems.append("time.understand.clamp_vs_cover.high must be above low")
    cap = u.get("news_cap_vs_cover")
    if not _is_num(cap) or cap < 1:
        problems.append("time.understand.news_cap_vs_cover must be a number, 1 or more")


def _check_rounding_and_buckets(t: dict, problems: list[str]) -> None:
    rn = t.get("round_nice")
    if not isinstance(rn, list) or not rn:
        problems.append("time.round_nice must be a non-empty list")
    else:
        belows = []
        for n, step in enumerate(rn):
            if not isinstance(step, dict) or not _is_num(step.get("step")) or step["step"] <= 0:
                problems.append(f"time.round_nice[{n}]: 'step' must be a number above 0")
                continue
            last = n == len(rn) - 1
            if last and "below" in step:
                problems.append("time.round_nice: the last entry must have no 'below' (it covers the rest)")
            if not last:
                if not _is_num(step.get("below")):
                    problems.append(f"time.round_nice[{n}]: needs a numeric 'below'")
                else:
                    belows.append(step["below"])
        if any(a >= b for a, b in zip(belows, belows[1:])):
            problems.append("time.round_nice: 'below' values must strictly increase")
    b = t.get("buckets")
    if not isinstance(b, dict):
        problems.append("time.buckets must be a mapping")
        return
    edges, labels = b.get("edges_minutes"), b.get("labels")
    if (not isinstance(edges, list) or not edges or any(not _is_num(e) or e <= 0 for e in edges)
            or any(a >= c for a, c in zip(edges, edges[1:]))):
        problems.append("time.buckets.edges_minutes must be a strictly increasing list of numbers above 0")
    elif not isinstance(labels, list) or len(labels) != len(edges) + 1:
        problems.append("time.buckets.labels must have one more entry than edges_minutes")
    elif any(not _is_text(x) for x in labels) or len(set(labels)) != len(labels):
        problems.append("time.buckets.labels must be non-empty and unique")


def _check_summary(data: dict, problems: list[str]) -> None:
    s = _section(data, "summary", dict, problems)
    if s is None:
        return
    tw, hw = s.get("target_words"), s.get("hard_words")
    ok = all(isinstance(x, dict) and _is_num(x.get("min")) and _is_num(x.get("max")) for x in (tw, hw))
    if not ok:
        problems.append("summary.target_words and summary.hard_words need numbers 'min' and 'max'")
    elif not (0 < hw["min"] <= tw["min"] < tw["max"] <= hw["max"]):
        problems.append("summary: need 0 < hard.min <= target.min < target.max <= hard.max")
    if not isinstance(s.get("single_paragraph"), bool):
        problems.append("summary.single_paragraph must be true or false")
    for key in ("must_cover", "motives"):
        v = s.get(key)
        if not isinstance(v, list) or not v or any(not _is_text(x) for x in v):
            problems.append(f"summary.{key} must be a non-empty list of text")


# ---------------------------------------------------------------- loading

def check_file(path: Path | str = DEFAULT_PATH) -> tuple[dict | None, str | None, list[str]]:
    """Read and validate a file without raising. Returns (data, sha, problems)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, None, [f"cannot read {path}: {exc.strerror or exc}"]
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, None, [f"{path.name} is not valid YAML: {exc}"]
    return data, rubrics_sha(raw), validate_rubrics(data)


def load_rubrics(path: Path | str = DEFAULT_PATH) -> Rubrics:
    """Load the rubrics or raise RubricsError listing every problem."""
    data, sha, problems = check_file(path)
    if problems or data is None or sha is None:     # the None checks only help the type checker
        raise RubricsError("rubrics problems:\n- " + "\n- ".join(problems))
    return Rubrics(data=data, sha=sha, path=Path(path))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_PATH
    data, sha, problems = check_file(path)
    if problems or data is None or sha is None:     # the None checks only help the type checker
        print(f"PROBLEMS in {path}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    weights = sum(a["weight"] for a in data["alignment"])
    print(f"OK: {path.name} (version {data['version']}), sha256 {sha[:12]}")
    print(f"  {len(data['difficulty'])} difficulty levels, {len(data['content_type'])} content types "
          f"(catch-all last), {len(data['alignment'])} alignment criteria (weights sum {weights:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
