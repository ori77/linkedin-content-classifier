"""Mutation check for pipeline/classifier.py (Step 9). Break one line at a time; tests must notice.
Run:  python tools/mutation_check_classifier.py        (exit 0 = all killed)

Not listed on purpose (no test could ever tell the difference):
  * `not repairs` inside first_try: a repair only happens after a retry, which already makes it False.
  * the upper clamp of the confidence value: the weights add up to exactly 1.
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    # helpers
    ("normalize_text: keep whitespace", 'return " ".join(text.split())', "return text"),
    ("input_hash: parts glued together", 'return hashlib.sha256("\\x1f".join(parts)', 'return hashlib.sha256("".join(parts)'),
    ("estimate_tokens: wrong ratio", "return len(text) // 4", "return len(text) // 5"),
    ("looks_cut_off: marker may be anywhere", r"\s*(see more|more)\s*$", r"\s*(see more|more)"),
    # prompts
    ("prompt A: keywords not capped", "s.keywords[:6]", "s.keywords"),
    # checking
    ("check_a: any subtopic accepted", 'if answer.get("subtopic_id") not in taxonomy.choice_ids:', "if False:"),
    ("check_b: key_points length unchecked", "not 1 <= len(kp) <= 5", "not 1 <= len(kp) <= 50"),
    ("check_b: zero minutes accepted", "if answer[k] < 1:", "if answer[k] < 0:"),
    ("check_b: subscore above 10 accepted", "if not 0 <= v <= 10 or v != int(v):", "if not 0 <= v <= 100 or v != int(v):"),
    ("check_b: fractional subscore accepted", " or v != int(v)", ""),
    ("check_b: self_confidence range", 'if not 0 <= answer["self_confidence"] <= 1:', 'if not 0 <= answer["self_confidence"] <= 10:'),
    ("summary: too short accepted", 'if n < s["hard_words"]["min"]:', "if n < 1:"),
    ("summary: too long accepted", 'elif n > s["hard_words"]["max"]:', "elif n > 10**6:"),
    ("summary: single line break accepted", 'if "\\n" in summary or "\\r" in summary:', 'if "\\n\\n" in summary:'),
    ("summary: bullets accepted", "if _BULLET_START.search(summary):", "if False:"),
    ("basis: 'stated' without a quote accepted", 'if answer["basis"] == "stated_in_post" and not has_size:', "if False:"),
    ("basis: quote without 'stated' accepted", 'if answer["basis"] != "stated_in_post" and has_size:', "if False:"),
    ("check_b: unknown stated_unit accepted", '("difficulty", DIFFICULTY_IDS), ("stated_unit", STATED_UNIT_IDS)):', '("difficulty", DIFFICULTY_IDS)):'),
    ("check_b: stated_count not a number accepted", '"self_confidence", "stated_count"):', '"self_confidence"):'),
    ("check_b: negative stated_count accepted", 'if answer["stated_count"] < 0:', "if False:"),
    ("prompt B: unit descriptions not shown", "{u['id']}: {u['describe']}", "{u['id']}: {u['id']}"),
    ("schema B: stated_unit not restricted", '"stated_unit": {"type": "string", "enum": list(STATED_UNIT_IDS)},', '"stated_unit": {"type": "string"},'),
    ("schema B: stated_count may be negative", '"stated_count": {"type": "number", "minimum": 0},', '"stated_count": {"type": "number"},'),
    # repairs
    ("repair: negative stated_count kept", 'if a["stated_count"] < 0:', "if False:"),
    ("repair: number without the post's words kept", 'if not has_size and (a["stated_count"], a["stated_unit"]) != (0, "none"):', "if False:"),
    ("repair: dropped stated_size keeps its number", '        a["stated_count"], a["stated_unit"] = 0, "none"\n        notes.append("repaired:stated_size")', '        notes.append("repaired:stated_size")'),
    ("repair: unusable stated size kept as stated", "        if why:", "        if False:"),
    ("stated check: unconvertible size not detected", 'if scoring.stated_minutes(a["stated_count"], a["stated_unit"], rub) is None:', "if False:"),
    ("stated check: wrong count not detected", 'if digits and a["stated_count"] not in digits:', "if False:"),
    ("stated check: only the first number compared", 'a["stated_count"] not in digits', 'a["stated_count"] != digits[0]'),
    ("stated check: decimal comma not understood", 'float(n.replace(",", "."))', "float(n)"),
    ("repair: unsupported 'stated' kept as estimated", 'a["basis"] = "post_only"                     # an unsupported claim', 'a["basis"] = "estimated_from_post"                     # an unsupported claim'),
    ("repair: stray stated_size kept", '        a["stated_size"] = ""\n', "        pass\n"),
    ("repair: subscore not capped at 10", "fixed = int(min(10, max(0, round(v))))", "fixed = int(max(0, round(v)))"),
    ("repair: self_confidence not clamped", 'c = min(1.0, max(0.0, float(a["self_confidence"])))', 'c = float(a["self_confidence"])'),
    ("repair: long summary not cut", "if len(words) > hard_max:", "if False:"),
    ("repair: cut ignores sentence ends", 'if kept[i].endswith((".", "!", "?")):', "if False:"),
    # coherence and confidence
    ("coherence: full project limit", 'time["understand_minutes"] < 180', 'time["understand_minutes"] < 18'),
    ("coherence: news hardness", 'difficulty in ("advanced", "expert")', 'difficulty in ("expert",)'),
    ("coherence: only the first number checked", "any(n not in post_numbers for n in size_numbers)", "size_numbers[0] not in post_numbers"),
    ("coherence: short summary limit", 'elif n < rub.data["summary"]["hard_words"]["min"]:', "elif n < 1:"),
    ("agreement: same subtopic not rewarded", "if chosen == clustering:", "if False:"),
    ("agreement: any topic counts as same", "taxonomy.topic_of(chosen) == taxonomy.topic_of(clustering)", "True"),
    ("confidence: agreement ignored", 'c["w_self"] * self_confidence + c["w_agree"] * agree + c["w_valid"] * c_valid', 'c["w_self"] * self_confidence + c["w_valid"] * c_valid'),
    ("confidence: no rescale without opinion", "if agree is None:", "if False:"),
    ("confidence: retry not penalised", 'c["valid_first_try"] if first_try else c["valid_after_retry"]', 'c["valid_first_try"]'),
    ("confidence: partial text not penalised", 'c["partial_factor"] if partial else 1.0', "1.0"),
    ("confidence: warnings not penalised", 'c["warning_penalty"] * penalties', "0"),
    ("confidence: can go negative", "min(1.0, max(0.0, value))", "min(1.0, value)"),
    ("needs_review: boundary", 'confidence < CONFIDENCE["review_below"]', 'confidence <= CONFIDENCE["review_below"]'),
    ("needs_review: other.general not reviewed", " or subtopic_id == OTHER_ID", ""),
    # cache key and context
    ("cache_key: post id left out", "str(post.activity_id), ", ""),
    ("cache_key: settings left out", "ctx.options_sig, ", ""),
    ("cache_key: taxonomy left out", "ctx.taxonomy.sha, ", ""),
    ("cache_key: rubrics left out", ", ctx.rubrics.sha]", "]"),
    ("cache_key: model left out", "ctx.prompt_sha, ctx.model, ", "ctx.prompt_sha, "),
    ("prompt_sha: user prompts left out", '"user_a": user_prompt_a(_DUMMY), ', ""),
    # asking the model
    ("ask: no retry", "for attempt in (1, 2):", "for attempt in (1,):"),
    ("ask: retry without the reason", "YOUR PREVIOUS ANSWER WAS REJECTED because: ", ""),
    ("ask: Ollama missing is retried", "except OllamaUnavailable:\n            raise\n", "except OllamaUnavailable:\n            answer, hard, soft = None, ['x'], []\n"),
    ("ask: retry reason overwritten by the retry", "        if attempt == 1:\n            first_problems = hard + soft", "        first_problems = hard + soft"),
    ("ask: retry reason never recorded", "            first_problems = hard + soft", "            first_problems = []"),
    ("ask: usable first answer thrown away", "if attempt == 1 and not hard:", "if False:"),
    # classify_post
    ("classify: topic retry not counted", "tries_a == 1 and tries_b == 1", "True and tries_b == 1"),
    ("classify: cut-off text not partial", "post.truncated or looks_cut_off(post.text)", "post.truncated"),
    ("classify: long post not partial", "post.truncated or looks_cut_off(post.text)", "looks_cut_off(post.text)"),
    ("classify: model's cover used for a stated size", 'if fixed["basis"] == "stated_in_post" else None)', "if False else None)"),
    ("classify: stated count not stored", '    time["stated_count"] = fixed["stated_count"]\n', "    pass\n"),
    ("classify: stated unit not stored", '    time["stated_unit"] = fixed["stated_unit"]\n', "    pass\n"),
    ("classify: retry reasons not stored", '"retry_reasons": {"topic": why_a, "rest": why_b},', '"retry_reasons": {"topic": [], "rest": []},'),
    ("classify: repairs not reported", '"warnings": warnings + repairs,', '"warnings": warnings,'),
    ("classify: missing opinion not reported", "if clustering is None:\n        warnings.append", "if False:\n        warnings.append"),
    ("classify: model's time rationale lost", '"time": fixed["time_reason"]', '"time": ""'),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/classifier.py", "tests/test_classifier.py", MUTATIONS))
