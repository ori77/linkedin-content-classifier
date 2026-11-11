"""The 7-lens classifier for one post (Step 9).

In simple terms, for each post we ask the local model TWO small questions, then do the
arithmetic ourselves:

  Call A  "which subtopic?"      multiple choice: the answer must be one of the 37 subtopic ids
                                 (or other.general). Lenses 1 and 2.
  Call B  "everything else"      content type, difficulty, time guesses, six alignment
                                 subscores, one-paragraph summary. Lenses 3 to 7.

Then plain code (scoring.py) turns the model's guesses into the final numbers, checks the
answer for mistakes (one retry with the mistake pointed out), repairs small slips, and
computes a confidence score. Nothing here talks to the network directly: the function that
asks the model is passed in, so tests can use a scripted fake.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import scoring
from classify_inputs import OTHER_ID, Post, Taxonomy
from ollama_json import OLLAMA_URL, LLMError, OllamaUnavailable, chat_json
from rubrics import (ALIGNMENT_IDS, BASIS_IDS, CONTENT_TYPE_IDS, DIFFICULTY_IDS, STATED_UNIT_IDS,
                     Rubrics)

PROMPT_VERSION = "classify-2026.11.3"       # a human label; the real guard is prompt_sha below

# Confidence (LLD 9.5, reworked: there is no embedding shortlist, so the "agreement" part compares
# the model's subtopic with the one clustering gave the post). Starting values, to be calibrated
# on the gold set in Step 10.
CONFIDENCE = {
    "w_self": 0.45, "w_agree": 0.35, "w_valid": 0.20,
    "agree_same_subtopic": 1.0, "agree_same_topic": 0.7, "agree_other": 0.4,
    "valid_first_try": 1.0, "valid_after_retry": 0.7,
    "partial_factor": 0.7, "warning_penalty": 0.1, "review_below": 0.60,
}
# Warnings that lower confidence (repairs are not here: a retry already lowers c_valid)
PENALISED_WARNINGS = {"full_project_short_understand", "news_hard", "stated_size_not_in_post",
                      "summary_short", "summary_missing"}

_INTERVIEW = re.compile(r"\binterview", re.IGNORECASE)    # also matches interviews, interviewer
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_CUT_OFF = re.compile(r"(…|\.\.\.)\s*(see more|more)\s*$", re.IGNORECASE)
_BULLET_START = re.compile(r"^\s*(?:[•·▪◦‣\-*–—]|\d+[.)])\s+", re.MULTILINE)


class ClassifyFailed(Exception):
    """The model never produced a usable answer for this post (after one retry)."""


# ------------------------------------------------------------------ small helpers

def normalize_text(text: str) -> str:
    return " ".join(text.split())


def input_hash(post: Post) -> str:
    """Fingerprint of everything about the post that the model sees."""
    parts = [normalize_text(post.text), post.attachment_title, post.attachment_site, post.links]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough size check only (about 4 characters per token)."""
    return len(text) // 4


def looks_cut_off(text: str) -> bool:
    """LinkedIn cards end with '...see more' when the post was not fully expanded."""
    return bool(_CUT_OFF.search(text.strip()))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


# ------------------------------------------------------------------ prompts

def system_prompt_a(taxonomy: Taxonomy, rubrics: Rubrics) -> str:
    lines = []
    for s in taxonomy.subtopics:
        kw = f" (typical words: {', '.join(s.keywords[:6])})" if s.keywords else ""
        lines.append(f"- {s.id} | {s.label} | {s.description}{kw}")
    return (
        "You are a meticulous librarian and senior AI Engineer. You file saved LinkedIn posts "
        f"for a person whose goal is: {rubrics.data['role_profile']}.\n\n"
        "TASK: choose the ONE subtopic below that best matches the main SUBJECT of the post "
        "(what it is about), not its format (course, roadmap, interview list, thread) and not "
        "the author's fame or hype.\n\n"
        "RULES\n"
        "1. Answer with a subtopic id from the list, copied exactly.\n"
        "2. Prefer the most specific subtopic that fits.\n"
        f"3. Use {OTHER_ID} only if no subtopic fits the main subject.\n"
        "4. evidence: one short sentence naming the concrete subject you saw in the post.\n\n"
        "SUBTOPICS (id | topic > subtopic | what it covers)\n" + "\n".join(lines) + "\n"
        f"- {OTHER_ID} | Other | Posts that fit no subtopic clearly.\n"
    )


def post_block(post: Post) -> str:
    out = f"POST\n<<<\n{post.text}\n>>>\n"
    if post.truncated:
        out += "NOTE: the post was very long and has been cut.\n"
    if post.attachment_title or post.attachment_site:
        out += f"LINK PREVIEW: {post.attachment_title or '(no title)'} ({post.attachment_site or 'unknown site'})\n"
    if post.links:
        out += f"LINKS IN POST: {post.links}\n"
    return out


def user_prompt_a(post: Post) -> str:
    return post_block(post)


def schema_a(taxonomy: Taxonomy) -> dict:
    return {
        "type": "object",
        "properties": {
            "evidence": {"type": "string"},
            "subtopic_id": {"type": "string", "enum": taxonomy.choice_ids},
        },
        "required": ["evidence", "subtopic_id"],
        "additionalProperties": False,
    }


def system_prompt_b(rubrics: Rubrics) -> str:
    r = rubrics.data
    t = r["time"]
    summ = r["summary"]
    parts = [
        "You are a meticulous librarian and senior AI Engineer. You classify saved LinkedIn posts "
        f"for a person whose goal is: {r['role_profile']}.\n",
        "WHAT YOU JUDGE\n" + " ".join(r["judging_basis"].split()) + "\n",
        "Answer ONLY with the requested JSON. Base every judgement on the CONTENT of the post, "
        "not on hype, likes or the author's fame. Keep every *_reason to one short sentence "
        "that names something concrete from the post.\n",
        "KEY POINTS: 1 to 5 short bullets of what the post teaches, claims or advertises.\n",
        "STATED SIZE AND BASIS (decide these before the time fields)\n"
        "- stated_size: copy the exact words of the post that give the size or duration of the "
        "resource (for example \"8-week course\", \"30 interview questions\", \"5-hour video\"). "
        "Leave it empty if the post gives none. A list of features (mentorship, mock interviews, "
        "a job board) is NOT a size.\n"
        "- stated_count: the number in stated_size, written as a number (8, 30, 5). Use 0 if "
        "stated_size is empty. Do NOT multiply or convert anything: the program does the arithmetic.\n"
        "- stated_unit, the unit of that number, one of:\n"
        + "\n".join(f"  * {u['id']}: {u['describe']}" for u in t["stated_units"]) + "\n"
        "- basis, one of:\n" + "\n".join(f"  * {b['id']}: {' '.join(b['rule'].split())}" for b in t["basis"]) + "\n",
        "CONTENT TYPE (use the FIRST rule that applies, from top to bottom)\n"
        "Check the rules in order, starting with the first, and stop at the first one the post fits. "
        "Do not skip an earlier rule because the post also looks like a later one: for example a post "
        "that asks an interview question and then explains the answer is an interview type, not a tutorial.\n"
        + "\n".join(f"- {c['id']}: {' '.join(c['rule'].split())}" for c in r["content_type"]) + "\n",
        "DIFFICULTY\n" + (" ".join(r["difficulty_guidance"].split()) + "\n" if r.get("difficulty_guidance") else "")
        + "\n".join(f"- {d['id']}: {d['definition']} For example: {d['example']}"
                    for d in r["difficulty"]) + "\n",
        "TIME (whole minutes)\n"
        "- cover_minutes: time to go through the whole resource once (read, watch, work through; "
        "run the code if it is a tutorial or project).\n"
        "- understand_minutes: time for a learner at the stated difficulty to really understand it "
        "and be able to reuse it (practice, re-reading, building). Never less than cover_minutes.\n"
        "- If basis is stated_in_post, the program works out the cover time itself from stated_count "
        "and stated_unit, so do no sums: just give your best figure. Never invent a size the post "
        "does not state.\n"
        "- If basis is estimated_from_post, estimate the build or tutorial from what the post describes.\n"
        "- If basis is post_only, cover_minutes should be close to the BASELINE given with the post "
        "(the reading time of the post itself).\n",
        "AI ENGINEER ALIGNMENT: score each criterion with a whole number from 0 to 10 using these anchors\n"
        + "\n".join(f"- {c['id']}: {c['question']} (10 = {c['high']}; 0 = {c['low']})" for c in r["alignment"]) + "\n",
        f"SUMMARY: exactly one paragraph of {summ['target_words']['min']} to {summ['target_words']['max']} "
        "words in plain language, with no bullet points and no line breaks. It must cover: "
        + " and ".join(summ["must_cover"]) + ". For the motive choose what fits: "
        + ", ".join(summ["motives"]) + ".\n",
        "SELF CONFIDENCE: a number from 0 to 1 for how sure you are of the whole answer; "
        "lower it if the post is vague, very short or cut off.\n",
    ]
    return "\n".join(parts)


def user_prompt_b(post: Post, baseline_minutes: int) -> str:
    return post_block(post) + f"\nBASELINE (reading time of the post itself): {baseline_minutes} minutes\nWORDS IN POST: {post.word_count}\n"


def schema_b(rubrics: Rubrics) -> dict:
    lim = rubrics.data["time"]["limits_minutes"]
    sub = {i: {"type": "integer", "minimum": 0, "maximum": 10} for i in ALIGNMENT_IDS}
    props = {
        "key_points": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "stated_size": {"type": "string"},
        "stated_count": {"type": "number", "minimum": 0},
        "stated_unit": {"type": "string", "enum": list(STATED_UNIT_IDS)},
        "basis": {"type": "string", "enum": list(BASIS_IDS)},
        "content_type": {"type": "string", "enum": list(CONTENT_TYPE_IDS)},
        "content_type_reason": {"type": "string"},
        "difficulty": {"type": "string", "enum": list(DIFFICULTY_IDS)},
        "difficulty_reason": {"type": "string"},
        "cover_minutes": {"type": "integer", "minimum": 1, "maximum": lim["cover_max"]},
        "understand_minutes": {"type": "integer", "minimum": 1, "maximum": lim["understand_max"]},
        "time_reason": {"type": "string"},
        "alignment": {"type": "object", "properties": sub, "required": list(ALIGNMENT_IDS),
                      "additionalProperties": False},
        "alignment_reason": {"type": "string"},
        "summary": {"type": "string"},
        "self_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


# ------------------------------------------------------------------ checking the answers

def check_a(answer: Any, taxonomy: Taxonomy) -> tuple[list[str], list[str]]:
    """Return (hard problems, soft problems). Hard = unusable; soft = fixable by repair."""
    if not isinstance(answer, dict):
        return ["the answer must be a JSON object"], []
    hard = []
    if answer.get("subtopic_id") not in taxonomy.choice_ids:
        hard.append(f"subtopic_id {answer.get('subtopic_id')!r} is not one of the listed ids")
    if not isinstance(answer.get("evidence"), str):
        hard.append("evidence must be text")
    return hard, []


def _summary_problems(summary: str, rub: Rubrics) -> list[str]:
    s = rub.data["summary"]
    out = []
    if "\n" in summary or "\r" in summary:
        out.append("summary must be ONE paragraph with no line breaks")
    if _BULLET_START.search(summary):
        out.append("summary must not use bullet points")
    n = len(summary.split())
    if n < s["hard_words"]["min"]:
        out.append(f"summary has {n} words, at least {s['hard_words']['min']} are needed "
                   f"(aim for {s['target_words']['min']} to {s['target_words']['max']})")
    elif n > s["hard_words"]["max"]:
        out.append(f"summary has {n} words, at most {s['hard_words']['max']} are allowed "
                   f"(aim for {s['target_words']['min']} to {s['target_words']['max']})")
    return out


def check_b(answer: Any, rub: Rubrics) -> tuple[list[str], list[str]]:
    if not isinstance(answer, dict):
        return ["the answer must be a JSON object"], []
    hard: list[str] = []
    soft: list[str] = []
    text_fields = ("stated_size", "content_type_reason", "difficulty_reason", "time_reason",
                   "alignment_reason", "summary")
    for k in text_fields:
        if not isinstance(answer.get(k), str):
            hard.append(f"{k} must be text")
    kp = answer.get("key_points")
    if not isinstance(kp, list) or not 1 <= len(kp) <= 5 or any(not isinstance(x, str) for x in kp):
        hard.append("key_points must be a list of 1 to 5 texts")
    for k, allowed in (("basis", BASIS_IDS), ("content_type", CONTENT_TYPE_IDS),
                       ("difficulty", DIFFICULTY_IDS), ("stated_unit", STATED_UNIT_IDS)):
        if answer.get(k) not in allowed:
            hard.append(f"{k} {answer.get(k)!r} is not one of {list(allowed)}")
    for k in ("cover_minutes", "understand_minutes", "self_confidence", "stated_count"):
        if not _is_num(answer.get(k)):
            hard.append(f"{k} must be a number")
    al = answer.get("alignment")
    if not isinstance(al, dict) or sorted(al) != sorted(ALIGNMENT_IDS) or any(not _is_num(v) for v in al.values()):
        hard.append(f"alignment must have exactly the numeric keys {list(ALIGNMENT_IDS)}")
    if hard:
        return hard, soft

    for k in ("cover_minutes", "understand_minutes"):
        if answer[k] < 1:
            soft.append(f"{k} must be at least 1")
    for k, v in al.items():
        if not 0 <= v <= 10 or v != int(v):
            soft.append(f"alignment.{k} must be a whole number from 0 to 10")
    if not 0 <= answer["self_confidence"] <= 1:
        soft.append("self_confidence must be from 0 to 1")
    if answer["stated_count"] < 0:
        soft.append("stated_count must be 0 or more")
    soft.extend(_summary_problems(answer["summary"], rub))
    has_size = answer["stated_size"].strip() != ""
    if answer["basis"] == "stated_in_post" and not has_size:
        soft.append("basis is stated_in_post but stated_size is empty: copy the post's words, or choose another basis")
    if answer["basis"] != "stated_in_post" and has_size:
        soft.append("stated_size is filled in but basis is not stated_in_post")
    return hard, soft


def _stated_problem(a: dict, rub: Rubrics) -> str | None:
    """Why a stated size cannot be used for the time sum (None = it can be used).

    "unconvertible": the unit is "other" (courses, modules...) or "none", or the number is 0.
    "count_mismatch": the post's words contain digits but the number the model reports is not
    one of them (for example words "20 questions" but count 200). Number words ("eight weeks")
    have no digits to compare, so they are accepted as given.
    """
    if scoring.stated_minutes(a["stated_count"], a["stated_unit"], rub) is None:
        return "unconvertible"
    digits = [float(n.replace(",", ".")) for n in _NUMBER.findall(a["stated_size"])]
    if digits and a["stated_count"] not in digits:
        return "count_mismatch"
    return None


def repair_b(answer: dict, rub: Rubrics) -> tuple[dict, list[str]]:
    """Fix small slips deterministically. Returns (fixed copy, list of what was changed)."""
    a = json.loads(json.dumps(answer))             # deep copy
    notes: list[str] = []
    for k, lo in (("cover_minutes", 1), ("understand_minutes", 1)):
        if a[k] < lo:
            a[k] = lo
            notes.append(f"repaired:{k}")
        a[k] = int(round(a[k]))
    for k, v in list(a["alignment"].items()):
        fixed = int(min(10, max(0, round(v))))
        if fixed != v:
            a["alignment"][k] = fixed
            notes.append(f"repaired:alignment.{k}")
    c = min(1.0, max(0.0, float(a["self_confidence"])))
    if c != a["self_confidence"]:
        a["self_confidence"] = c
        notes.append("repaired:self_confidence")
    if a["stated_count"] < 0:
        a["stated_count"] = 0
        notes.append("repaired:stated_count")
    has_size = a["stated_size"].strip() != ""
    if not has_size and (a["stated_count"], a["stated_unit"]) != (0, "none"):
        a["stated_count"], a["stated_unit"] = 0, "none"      # a number without the post's words behind it
        notes.append("repaired:stated_count")
    if a["basis"] == "stated_in_post" and not has_size:
        a["basis"] = "post_only"                     # an unsupported claim falls back to the safe reading
        notes.append("repaired:basis")
    elif a["basis"] != "stated_in_post" and has_size:
        a["stated_size"] = ""
        a["stated_count"], a["stated_unit"] = 0, "none"
        notes.append("repaired:stated_size")
    if a["basis"] == "stated_in_post":
        why = _stated_problem(a, rub)
        if why:                                      # the size cannot be turned into minutes safely
            a["basis"] = "post_only"                 # (the words and number stay in the record for audit)
            notes.append(f"repaired:basis_{why}")
    summary = a["summary"]
    no_bullets = _BULLET_START.sub("", summary.replace("\r", ""))      # bullets first, line by line
    cleaned = re.sub(r"\s*\n+\s*", " ", no_bullets).strip()               # then one single line
    if cleaned != summary:
        a["summary"] = cleaned
        notes.append("repaired:summary_format")
    words = a["summary"].split()
    hard_max = rub.data["summary"]["hard_words"]["max"]
    if len(words) > hard_max:
        kept = words[:hard_max]
        # prefer to stop at the last full sentence if that keeps a reasonable amount
        for i in range(len(kept) - 1, rub.data["summary"]["target_words"]["min"] - 1, -1):
            if kept[i].endswith((".", "!", "?")):
                kept = kept[: i + 1]
                break
        a["summary"] = " ".join(kept)
        notes.append("repaired:summary_trimmed")
    return a, notes


# ------------------------------------------------------------------ coherence and confidence

def coherence_warnings(*, content_type: str, difficulty: str, time: dict, stated_size: str,
                       post_text: str, summary: str, rub: Rubrics) -> list[str]:
    """Things that look inconsistent (LLD 7.8, 'type/time coherence')."""
    w = []
    if content_type == "full_project" and time["understand_minutes"] < 180:
        w.append("full_project_short_understand")
    if content_type == "news" and difficulty in ("advanced", "expert"):
        w.append("news_hard")
    if stated_size.strip():
        post_numbers = set(_NUMBER.findall(post_text))
        size_numbers = _NUMBER.findall(stated_size)
        if size_numbers and any(n not in post_numbers for n in size_numbers):
            w.append("stated_size_not_in_post")
    n = len(summary.split())
    s = rub.data["summary"]
    if n == 0:
        w.append("summary_missing")
    elif n < s["hard_words"]["min"]:
        w.append("summary_short")
    elif n < s.get("brief_words_below", 0):            # v2 only: accepted, but shorter than the aim
        w.append("summary_brief")
    if content_type != "interview_question_set" and _INTERVIEW.search(post_text):
        w.append("interview_mentioned")                # a prompt for a human glance; never lowers confidence
    return w


def agreement(chosen: str, clustering: str | None, taxonomy: Taxonomy) -> float | None:
    """How much the model agrees with the subtopic clustering gave the post (None = no opinion)."""
    if clustering is None:
        return None
    if chosen == clustering:
        return CONFIDENCE["agree_same_subtopic"]
    # "other.general" belongs to the reserved topic "other", which no real subtopic shares
    if taxonomy.topic_of(chosen) == taxonomy.topic_of(clustering):
        return CONFIDENCE["agree_same_topic"]
    return CONFIDENCE["agree_other"]


def compute_confidence(*, self_confidence: float, agree: float | None, first_try: bool,
                       partial: bool, penalties: int) -> tuple[float, dict]:
    """LLD 9.5 reworked. Returns (confidence 0-1 with 2 decimals, the parts for auditing)."""
    c = CONFIDENCE
    c_valid = c["valid_first_try"] if first_try else c["valid_after_retry"]
    if agree is None:       # no clustering opinion: leave that part out and rescale the others
        base = (c["w_self"] * self_confidence + c["w_valid"] * c_valid) / (c["w_self"] + c["w_valid"])
    else:
        base = c["w_self"] * self_confidence + c["w_agree"] * agree + c["w_valid"] * c_valid
    value = base * (c["partial_factor"] if partial else 1.0) - c["warning_penalty"] * penalties
    value = round(min(1.0, max(0.0, value)) + 1e-9, 2)
    return value, {"self": self_confidence, "agree": agree, "valid": c_valid,
                   "partial": partial, "penalties": penalties}


def needs_review(confidence: float, subtopic_id: str) -> bool:
    return confidence < CONFIDENCE["review_below"] or subtopic_id == OTHER_ID


# ------------------------------------------------------------------ context and cache key

_DUMMY = Post(activity_id=0, text="dummy post text", attachment_title="t", attachment_site="s",
              links="l", truncated=True)


@dataclass
class Context:
    """Everything that stays the same for a whole run."""
    model: str
    taxonomy: Taxonomy
    rubrics: Rubrics
    assignments: dict[int, str]
    url: str = OLLAMA_URL
    num_ctx: int = 8192
    think: bool = False
    options: dict | None = None
    system_a: str = field(init=False)
    system_b: str = field(init=False)
    schema_a: dict = field(init=False)
    schema_b: dict = field(init=False)
    prompt_sha: str = field(init=False)
    options_sig: str = field(init=False)

    def __post_init__(self):
        self.system_a = system_prompt_a(self.taxonomy, self.rubrics)
        self.system_b = system_prompt_b(self.rubrics)
        self.schema_a = schema_a(self.taxonomy)
        self.schema_b = schema_b(self.rubrics)
        # Any change to a prompt, a schema or the way the post is shown changes this hash,
        # so old answers are never reused with different prompts.
        blob = json.dumps({"version": PROMPT_VERSION, "a": self.system_a, "b": self.system_b,
                           "schema_a": self.schema_a, "schema_b": self.schema_b,
                           "user_a": user_prompt_a(_DUMMY), "user_b": user_prompt_b(_DUMMY, 1)},
                          sort_keys=True)
        self.prompt_sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        self.options_sig = json.dumps({"think": self.think, "num_ctx": self.num_ctx,
                                       "options": self.options or {}}, sort_keys=True)


def cache_key(post: Post, ctx: Context) -> str:
    """LLD 9.6: content | prompt | model | settings | taxonomy | rubrics, plus the post's own id.

    The id is added because reposts (identical text, different post) each need their own row.
    """
    parts = [str(post.activity_id), input_hash(post), ctx.prompt_sha, ctx.model, ctx.options_sig, ctx.taxonomy.sha, ctx.rubrics.sha]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ asking the model

ChatFn = Callable[..., tuple[Any, dict]]


def _ask(chat: ChatFn, ctx: Context, system: str, user: str, schema: dict, num_predict: int,
         check: Callable[[Any], tuple[list[str], list[str]]]) -> tuple[Any, int, list[str], dict, list[str]]:
    """Ask, check, and retry ONCE with the mistakes pointed out.

    Returns (answer, attempts, leftover soft problems, usage, what was wrong with the first
    answer). Raises ClassifyFailed when the second answer still has hard problems.
    OllamaUnavailable is never retried.
    """
    usage = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
    feedback = ""
    answer, hard, soft = None, ["no answer"], []
    usable_first = None                      # a first answer that only had fixable (soft) problems
    first_problems: list[str] = []           # why a retry was needed (stored, so we can study it)
    for attempt in (1, 2):
        try:
            answer, stats = chat(ctx.model, system, user + feedback, schema, url=ctx.url,
                                 num_ctx=ctx.num_ctx, think=ctx.think, num_predict=num_predict,
                                 options=ctx.options)
        except OllamaUnavailable:
            raise
        except LLMError as exc:
            answer, hard, soft = None, [str(exc)], []
        else:
            usage["seconds"] += stats.get("seconds") or 0
            usage["input_tokens"] += stats.get("prompt_tokens") or 0
            usage["output_tokens"] += stats.get("output_tokens") or 0
            hard, soft = check(answer)
        if not hard and not soft:
            return answer, attempt, [], usage, first_problems
        if attempt == 1:
            first_problems = hard + soft
        if attempt == 1 and not hard:
            usable_first = (answer, soft)
        feedback = ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED because: " + "; ".join(hard + soft)
                    + ". Answer again and fix this.")
    if hard and usable_first is not None:    # the retry broke something worse: keep the first answer
        return usable_first[0], 2, usable_first[1], usage, first_problems
    if hard:
        raise ClassifyFailed("; ".join(hard))
    return answer, 2, soft, usage, first_problems


def classify_post(post: Post, ctx: Context, chat: ChatFn = chat_json) -> dict:
    """Classify one post on all seven lenses. Returns the record to store."""
    baseline = scoring.baseline_cover_minutes(post.word_count, ctx.rubrics)

    ans_a, tries_a, _, use_a, why_a = _ask(chat, ctx, ctx.system_a, user_prompt_a(post), ctx.schema_a, 400,
                                    lambda a: check_a(a, ctx.taxonomy))
    ans_b, tries_b, _, use_b, why_b = _ask(chat, ctx, ctx.system_b, user_prompt_b(post, baseline),
                                    ctx.schema_b, 1500, lambda a: check_b(a, ctx.rubrics))
    fixed, repairs = repair_b(ans_b, ctx.rubrics)

    stated = (scoring.stated_minutes(fixed["stated_count"], fixed["stated_unit"], ctx.rubrics)
              if fixed["basis"] == "stated_in_post" else None)      # the program, not the model, multiplies
    time = scoring.finalize_time(
        baseline=baseline, llm_cover=fixed["cover_minutes"], llm_understand=fixed["understand_minutes"],
        difficulty=fixed["difficulty"], content_type=fixed["content_type"], basis=fixed["basis"],
        rub=ctx.rubrics, stated=stated)
    time["stated_size"] = fixed["stated_size"].strip()
    time["stated_count"] = fixed["stated_count"]
    time["stated_unit"] = fixed["stated_unit"]
    time["model_cover_minutes"] = fixed["cover_minutes"]
    time["model_understand_minutes"] = fixed["understand_minutes"]

    score = scoring.alignment_score(fixed["alignment"], ctx.rubrics)
    warnings = coherence_warnings(content_type=fixed["content_type"], difficulty=fixed["difficulty"],
                                  time=time, stated_size=fixed["stated_size"], post_text=post.text,
                                  summary=fixed["summary"], rub=ctx.rubrics)
    subtopic = ans_a["subtopic_id"]
    clustering = ctx.assignments.get(post.activity_id)
    partial = post.truncated or looks_cut_off(post.text)
    conf, parts = compute_confidence(
        self_confidence=fixed["self_confidence"], agree=agreement(subtopic, clustering, ctx.taxonomy),
        first_try=(tries_a == 1 and tries_b == 1 and not repairs), partial=partial,
        penalties=sum(1 for w in warnings if w in PENALISED_WARNINGS))
    if partial:
        warnings.append("text_partial")
    if clustering is None:
        warnings.append("no_clustering_opinion")

    return {
        "activity_id": post.activity_id,
        "topic_id": ctx.taxonomy.topic_of(subtopic),
        "subtopic_id": subtopic,
        "difficulty": fixed["difficulty"],
        "content_type": fixed["content_type"],
        "time": time,
        "alignment_score": score,
        "alignment_band": scoring.alignment_band(score, ctx.rubrics),
        "alignment_subscores": fixed["alignment"],
        "summary": fixed["summary"],
        "key_points": fixed["key_points"],
        "rationales": {"topic": ans_a["evidence"], "content_type": fixed["content_type_reason"],
                       "difficulty": fixed["difficulty_reason"], "time": fixed["time_reason"],
                       "alignment": fixed["alignment_reason"]},
        "confidence": conf,
        "confidence_parts": parts,
        "needs_review": needs_review(conf, subtopic),
        "warnings": warnings + repairs,
        "clustering_subtopic_id": clustering,
        "model": ctx.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": ctx.prompt_sha,
        "taxonomy_version": ctx.taxonomy.version,
        "rubrics_version": ctx.rubrics.version,
        "rubrics_sha": ctx.rubrics.sha,
        "cache_key": cache_key(post, ctx),
        "attempts": {"topic": tries_a, "rest": tries_b},
        "retry_reasons": {"topic": why_a, "rest": why_b},
        "input_tokens": use_a["input_tokens"] + use_b["input_tokens"],
        "output_tokens": use_a["output_tokens"] + use_b["output_tokens"],
        "seconds": round(use_a["seconds"] + use_b["seconds"], 1),
    }
