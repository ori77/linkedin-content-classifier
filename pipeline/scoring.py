"""Turn the model's judgements into final numbers (Step 9).

In simple terms: the model is good at judging ("this looks like an intermediate tutorial,
maybe 20 minutes") and bad at arithmetic and at keeping to rules. So the model only says what
it thinks, and THIS file does all the sums, using the numbers in rubrics.v1.yaml:

  * baseline_cover_minutes : how long the post itself takes to read (words / speed)
  * stated_minutes         : turns "20" + "questions" into 60 minutes (the model only reads the
                             number and the unit off the post; this file multiplies)
  * finalize_time          : keep the model's guess within the allowed limits, work out
                             "time to understand", round to tidy numbers, add the bucket
  * alignment_score        : weighted sum of the six 0-10 subscores -> one score and a band
  * round_nice / bucket    : the rounding and bucket rules

Nothing here calls a model or reads a file, so it is fast to test.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from rubrics import ALIGNMENT_IDS, BASIS_IDS, CONTENT_TYPE_IDS, DIFFICULTY_IDS, STATED_UNIT_IDS, Rubrics


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------ rounding and buckets

def round_nice(minutes: float, rub: Rubrics) -> int:
    """Round to a tidy number: LLD 9.2 (below 60 -> nearest 5, below 480 -> nearest 15, else 60).

    Never returns less than one step (a 2-minute read shows as 5, not 0).
    """
    if not _is_number(minutes) or minutes <= 0:
        raise ValueError(f"minutes must be a positive number, got {minutes!r}")
    for entry in rub.data["time"]["round_nice"]:
        if "below" not in entry or minutes < entry["below"]:
            step = entry["step"]
            break
    rounded = int(math.floor(minutes / step + 0.5)) * step   # halves go up (round() would not)
    return max(rounded, step)


def bucket(minutes: float, rub: Rubrics) -> str:
    """Label such as '15-60m'. An edge belongs to the bucket above it (15 -> '15-60m')."""
    b = rub.data["time"]["buckets"]
    return b["labels"][bisect_right(b["edges_minutes"], minutes)]


# ------------------------------------------------------------------ time

def baseline_cover_minutes(word_count: float, rub: Rubrics, technical: bool = False) -> int:
    """Reading time of the post text itself, in whole minutes, at least 1 (LLD 9.1)."""
    if not _is_number(word_count) or word_count < 0:
        raise ValueError(f"word_count must be a number, 0 or more, got {word_count!r}")
    t = rub.data["time"]
    wpm = t["technical_wpm"] if technical else t["reading_wpm"]
    return math.ceil(max(1.0, word_count / wpm))


def stated_minutes(count: float, unit: str, rub: Rubrics) -> float | None:
    """The size a post states, in minutes: 20 questions -> 60, 5 hours -> 300, 8 weeks -> 2400.

    Returns None when it cannot be worked out: the unit is "other" (courses, modules, ...) or
    "none", or the number is not above 0. The caller then falls back to basis post_only.
    """
    if unit not in STATED_UNIT_IDS:
        raise ValueError(f"unknown stated unit {unit!r}")
    if not _is_number(count) or count <= 0:
        return None
    for item in rub.data["time"]["stated_units"]:
        if item["id"] == unit:
            per = item.get("minutes_per_unit")
            return None if per is None else count * per
    raise ValueError(f"unit {unit!r} is missing from the rubrics")      # cannot happen: validator checks


def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))


def finalize_time(*, baseline: int, llm_cover: float, llm_understand: float, difficulty: str,
                  content_type: str, basis: str, rub: Rubrics,
                  stated: float | None = None) -> dict[str, Any]:
    """Combine the model's two time guesses with the rules. Returns the stored time estimate.

    How the rules depend on `basis` (the Step 8 decision to judge the advertised resource):
      * post_only            : the post is all we judge, so the model's cover time must stay
                               within 0.5x to 3x of the post's own reading time.
      * stated_in_post       : the resource is bigger than the post (a 60-hour course), so the
                               reading time of the post is irrelevant. `stated` (the stated
                               size in minutes, from stated_minutes) REPLACES the model's cover
                               guess, because the model is bad at sums. It is required for
                               this basis. The result is never below the hands-on floor of the
                               content type (a mini-project is at least 45 minutes) and never
                               above the absolute limit (14 days).
      * estimated_from_post  : same, but with the model's own estimate; `stated` must be None.
    """
    if difficulty not in DIFFICULTY_IDS:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    if content_type not in CONTENT_TYPE_IDS:
        raise ValueError(f"unknown content_type {content_type!r}")
    if basis not in BASIS_IDS:
        raise ValueError(f"unknown basis {basis!r}")
    for name, value in (("baseline", baseline), ("llm_cover", llm_cover),
                        ("llm_understand", llm_understand)):
        if not _is_number(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")

    if basis == "stated_in_post":
        if not _is_number(stated) or stated <= 0:
            raise ValueError("basis stated_in_post needs `stated` (the stated size in minutes, above 0)")
    elif stated is not None:
        raise ValueError(f"`stated` is only for basis stated_in_post, not {basis!r}")

    t = rub.data["time"]
    lim = t["limits_minutes"]
    hands_on = t["hands_on_floor_minutes"][content_type] if basis in t["hands_on_applies_when_basis"] else 0

    # --- time to cover
    if basis in t["cover_clamp_applies_when_basis"]:
        ref = baseline + hands_on
        low, high = t["cover_clamp"]["low"] * ref, t["cover_clamp"]["high"] * ref
    else:
        low, high = hands_on, lim["cover_max"]
    cover = _clamp(stated if stated is not None else llm_cover, low, high)
    cover = _clamp(cover, 1, lim["cover_max"])

    # --- time to understand: blend the model's guess with cover x depth x type
    u = t["understand"]
    heuristic = cover * u["depth_multiplier"][difficulty] * u["type_multiplier"][content_type]
    understand = math.sqrt(llm_understand * heuristic)               # geometric mean
    understand = _clamp(understand, cover * u["clamp_vs_cover"]["low"], cover * u["clamp_vs_cover"]["high"])
    if content_type == "news":                                       # nothing to "master" in news
        understand = max(cover, min(understand, cover * u["news_cap_vs_cover"]))
    understand = min(understand, lim["understand_max"])

    # --- tidy numbers; buckets come from the tidy numbers so the two always agree
    cover_r = round_nice(cover, rub)
    understand_r = max(round_nice(understand, rub), cover_r)
    return {
        "cover_minutes": cover_r,
        "understand_minutes": understand_r,
        "cover_bucket": bucket(cover_r, rub),
        "understand_bucket": bucket(understand_r, rub),
        "baseline_cover_minutes": int(baseline),
        "basis": basis,
        "stated_minutes": None if stated is None else round(stated, 2),
    }


# ------------------------------------------------------------------ alignment

def alignment_score(subscores: dict[str, int], rub: Rubrics) -> float:
    """Weighted sum of the six 0-10 subscores, rounded to one decimal (halves round up)."""
    if not isinstance(subscores, dict) or sorted(subscores) != sorted(ALIGNMENT_IDS):
        raise ValueError(f"subscores must have exactly the keys {ALIGNMENT_IDS}")
    total = 0.0
    for item in rub.data["alignment"]:
        s = subscores[item["id"]]
        if not _is_number(s) or not 0 <= s <= 10:
            raise ValueError(f"subscore {item['id']} must be a number from 0 to 10, got {s!r}")
        total += item["weight"] * s
    # round(total, 1) alone would turn 7.05 into 7.0 or 7.1 depending on float noise
    return float(Decimal(repr(round(total, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def alignment_band(score: float, rub: Rubrics) -> str:
    """Meaning of a score (LLD 9.4), e.g. 7.4 -> 'Highly relevant'."""
    if not _is_number(score) or not 0 <= score <= 10:
        raise ValueError(f"score must be from 0 to 10, got {score!r}")
    for band in rub.data["alignment_bands"]:          # highest band first
        if score >= band["min"]:
            return band["label"]
    raise ValueError("no band matched")                # cannot happen: the validator requires a band at 0
