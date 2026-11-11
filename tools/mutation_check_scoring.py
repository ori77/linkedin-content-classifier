"""Mutation check for pipeline/scoring.py (Step 9).

In simple terms: break the arithmetic one line at a time and make sure the tests notice.
Run:  python tools/mutation_check_scoring.py        (exit 0 = all killed, 1 = some survived)

Not listed on purpose (the break cannot change any result, so no test could ever notice):
  * `understand` lower clamp vs cover and the final `max(..., cover_r)`: two safety nets that
    both guarantee understand >= cover, so removing only one changes nothing.
  * the lower bound 1 on cover: round_nice already never returns less than one step.
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    # rounding and buckets
    ("round_nice: halves round to even", "int(math.floor(minutes / step + 0.5)) * step", "int(round(minutes / step)) * step"),
    ("round_nice: allow results below one step", "return max(rounded, step)", "return rounded"),
    ("round_nice: wrong band edge", 'minutes < entry["below"]', 'minutes < entry["below"] * 2'),
    ("round_nice: accept zero", "not _is_number(minutes) or minutes <= 0", "not _is_number(minutes) or minutes < 0"),
    ("bucket: edge belongs to the bucket below", 'bisect_right(b["edges_minutes"], minutes)', 'bisect_right(b["edges_minutes"], minutes - 1e-9)'),
    # baseline
    ("baseline: round down instead of up", "math.ceil(max(1.0, word_count / wpm))", "math.floor(max(1.0, word_count / wpm))"),
    ("baseline: allow 0 minutes", "math.ceil(max(1.0, word_count / wpm))", "math.ceil(max(0.0, word_count / wpm))"),
    ("baseline: technical speed ignored", 't["technical_wpm"] if technical else t["reading_wpm"]', 't["reading_wpm"] if technical else t["reading_wpm"]'),
    ("number check: accept True", "and not isinstance(x, bool)", ""),
    ("number check: accept NaN and infinity", " and math.isfinite(x)", ""),
    # finalize: cover
    ("cover: always clamp to the post's reading time", 'if basis in t["cover_clamp_applies_when_basis"]:', "if True:"),
    ("cover: hands-on floor always applies", 'if basis in t["hands_on_applies_when_basis"] else 0', "if True else 0"),
    ("cover: lower clamp removed", 't["cover_clamp"]["low"] * ref', "0 * ref"),
    ("cover: upper clamp removed", 't["cover_clamp"]["high"] * ref', "1e9 * ref"),
    ("cover: bucket from unrounded number", "bucket(cover_r, rub)", "bucket(cover, rub)"),
    ("clamp helper: swapped min and max", "return max(low, min(x, high))", "return min(low, max(x, high))"),
    # finalize: understand
    ("understand: depth multiplier ignored", 'u["depth_multiplier"][difficulty]', "1.0"),
    ("understand: type multiplier ignored", 'u["type_multiplier"][content_type]', "1.0"),
    ("understand: arithmetic mean instead of geometric", "math.sqrt(llm_understand * heuristic)", "(llm_understand + heuristic) / 2"),
    ("understand: upper clamp removed", 'cover * u["clamp_vs_cover"]["high"]', "1e12"),
    ("understand: news rule off", 'if content_type == "news":', "if False:"),
    ("understand: news cap too loose", 'cover * u["news_cap_vs_cover"]', "cover * 100"),
    ("understand: 30-day limit removed", 'understand = min(understand, lim["understand_max"])', "understand = understand"),
    ("finalize: accept unknown basis", "if basis not in BASIS_IDS:", "if False:"),
    ("finalize: accept unknown difficulty", "if difficulty not in DIFFICULTY_IDS:", "if False:"),
    ("finalize: accept bad numbers", "not _is_number(value) or value <= 0", "False"),
    # stated size -> minutes (Step 9 change)
    ("stated: model's cover guess used instead of the stated minutes", "stated if stated is not None else llm_cover", "llm_cover"),
    ("stated: not required for the stated basis", "if not _is_number(stated) or stated <= 0:", "if False:"),
    ("stated: allowed for other bases", "elif stated is not None:", "elif False:"),
    ("stated_minutes: zero count accepted", "if not _is_number(count) or count <= 0:", "if not _is_number(count) or count < 0:"),
    ("stated_minutes: unit ignored", "return None if per is None else count * per", "return count"),
    ("stated_minutes: unconvertible unit gets a number", "return None if per is None else count * per", "return count * (per or 1)"),
    ("stated_minutes: unknown unit accepted", "if unit not in STATED_UNIT_IDS:", "if False:"),
    # alignment
    ("alignment: weights ignored", 'total += item["weight"] * s', "total += s / 6"),
    ("alignment: plain round() instead of round-half-up", 'float(Decimal(repr(round(total, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))', "round(total, 1)"),
    ("alignment: allow subscores up to 100", "not 0 <= s <= 10", "not 0 <= s <= 100"),
    ("alignment: do not check the key names", "sorted(subscores) != sorted(ALIGNMENT_IDS)", "False"),
    ("band: boundary belongs to the band below", 'score >= band["min"]', 'score > band["min"]'),
    ("band: accept scores above 10", "not 0 <= score <= 10", "not 0 <= score <= 100"),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/scoring.py", "tests/test_scoring.py", MUTATIONS))
