"""Tests for pipeline/classifier.py (Step 9, part 2). A scripted fake model replaces Ollama."""
import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import classifier as C  # noqa: E402
import classify_inputs as I  # noqa: E402
import rubrics as R  # noqa: E402
from ollama_json import LLMError, OllamaUnavailable  # noqa: E402

RUB = R.load_rubrics(ROOT / "rubrics" / "rubrics.v1.yaml")

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "Retrieval-Augmented Generation", "description": "RAG systems.",
         "subtopics": [
             {"id": "rag.chunking", "name": "Chunking", "description": "How to split documents.",
              "keywords": ["chunk", "split"]},
             {"id": "rag.evaluation", "name": "RAG Evaluation", "description": "Measuring RAG."}]},
        {"id": "careers", "name": "Career Development", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
    ],
    "other": {"id": "other.general", "name": "Other / General", "description": "Fits nothing."},
}


@pytest.fixture(scope="module")
def tax(tmp_path_factory):
    p = tmp_path_factory.mktemp("tax") / "tax.yaml"
    p.write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    return I.load_taxonomy(p)


def make_ctx(tax, **kw):
    args = dict(model="m1", taxonomy=tax, rubrics=RUB, assignments={1: "rag.chunking"})
    args.update(kw)
    return C.Context(**args)


def words(n, sentence=8):
    """n words, in sentences of `sentence` words each ending with a full stop."""
    out = []
    for i in range(n):
        w = "word"
        if (i + 1) % sentence == 0:
            w += "."
        out.append(w)
    return " ".join(out)


def good_b(**over):
    b = {"key_points": ["explains chunking"], "stated_size": "", "stated_count": 0, "stated_unit": "none",
         "basis": "post_only",
         "content_type": "tutorial", "content_type_reason": "Teaches step by step.",
         "difficulty": "intermediate", "difficulty_reason": "Needs Python.",
         "cover_minutes": 2, "understand_minutes": 10, "time_reason": "Short post.",
         "alignment": {"applicability": 8, "technical_depth": 6, "hands_on": 9, "currency": 7,
                       "career_value": 5, "production_relevance": 4},
         "alignment_reason": "Useful.", "summary": words(60), "self_confidence": 0.8}
    b.update(over)
    return b


GOOD_A = {"evidence": "chunking documents", "subtopic_id": "rag.chunking"}
POST = I.Post(activity_id=1, text="Ten words about chunking documents for retrieval systems in practice.")


class FakeChat:
    """Plays back scripted answers. An Exception in the script is raised instead of returned."""

    def __init__(self, a=None, b=None):
        self.a = list(a if a is not None else [GOOD_A])
        self.b = list(b if b is not None else [good_b()])
        self.calls = []

    def __call__(self, model, system, user, schema, **kw):
        is_a = "evidence" in schema["properties"]
        self.calls.append({"kind": "a" if is_a else "b", "model": model, "system": system,
                           "user": user, "schema": schema, **kw})
        item = (self.a if is_a else self.b).pop(0)
        if isinstance(item, Exception):
            raise item
        stats = {"seconds": 1.5, "prompt_tokens": 100, "output_tokens": 20} if is_a else \
                {"seconds": 4.0, "prompt_tokens": 200, "output_tokens": 150}
        return copy.deepcopy(item), stats


# ============================================================ helpers

def test_normalize_text():
    assert C.normalize_text("  a \n b\t c ") == "a b c"


def test_input_hash_ignores_whitespace_but_not_content():
    a = C.input_hash(I.Post(1, "hello   world"))
    assert a == C.input_hash(I.Post(1, "hello world\n"))
    assert a != C.input_hash(I.Post(1, "hello there"))
    assert a != C.input_hash(I.Post(1, "hello world", attachment_title="t"))
    assert a != C.input_hash(I.Post(1, "hello world", attachment_site="s"))
    assert a != C.input_hash(I.Post(1, "hello world", links="l"))


def test_input_hash_parts_cannot_be_confused():
    # "ab" + "" must differ from "a" + "b": the parts are separated
    assert C.input_hash(I.Post(1, "x", attachment_title="ab", attachment_site="")) != \
           C.input_hash(I.Post(1, "x", attachment_title="a", attachment_site="b"))


@pytest.mark.parametrize("text,expected", [
    ("Great post about RAG …see more", True), ("Great post...see more", True),
    ("Great post… more", True), ("Great post …See More  ", True),
    ("Great post about RAG.", False), ("I want to see more of this", False),
    ("Great… see more of this thread tomorrow", False),        # the marker must be at the very end
])
def test_looks_cut_off(text, expected):
    assert C.looks_cut_off(text) is expected


def test_estimate_tokens():
    assert C.estimate_tokens("a" * 400) == 100


# ============================================================ prompts and schemas

def test_system_a_lists_every_choice(tax):
    s = C.system_prompt_a(tax, RUB)
    for sub in tax.subtopics:
        assert f"- {sub.id} | {sub.label} | {sub.description}" in s
    assert "other.general" in s and RUB.data["role_profile"] in s
    assert "(typical words: chunk, split)" in s
    assert "rag.evaluation | Retrieval-Augmented Generation > RAG Evaluation | Measuring RAG.\n" in s  # no empty parens


def test_system_a_shows_at_most_six_keywords(tax):
    sub = I.Subtopic(id="rag.many", name="Many", topic_id="rag", topic_name="RAG", description="d",
                     keywords=tuple(f"k{i}" for i in range(8)))
    big = I.Taxonomy(version="v1", subtopics=(sub,), sha="x", path=tax.path, by_id={sub.id: sub})
    s = C.system_prompt_a(big, RUB)
    assert "(typical words: k0, k1, k2, k3, k4, k5)" in s and "k6" not in s


def test_system_a_asks_for_subject_not_format(tax):
    s = C.system_prompt_a(tax, RUB)
    assert "SUBJECT" in s and "not its format" in s


def test_schema_a_enum_is_the_choice_list(tax):
    sch = C.schema_a(tax)
    assert sch["properties"]["subtopic_id"]["enum"] == tax.choice_ids
    assert sch["required"] == ["evidence", "subtopic_id"] and sch["additionalProperties"] is False
    assert list(sch["properties"]) == ["evidence", "subtopic_id"]       # evidence is written first


def test_system_b_contains_every_rule_from_the_rubrics():
    s = C.system_prompt_b(RUB)
    d = RUB.data
    assert " ".join(d["judging_basis"].split()) in s
    for item in d["difficulty"]:
        assert f"- {item['id']}: {item['definition']}" in s
    for item in d["alignment"]:
        assert f"- {item['id']}: {item['question']} (10 = {item['high']}; 0 = {item['low']})" in s
    for b in d["time"]["basis"]:
        assert f"* {b['id']}:" in s
    assert "60 to 150 words" in s
    for u in d["time"]["stated_units"]:
        assert f"* {u['id']}: {u['describe']}" in s
    assert "Do NOT multiply or convert anything" in s and "is NOT a size" in s
    assert "works out the cover time itself" in s
    for m in d["summary"]["motives"]:
        assert m in s


def test_system_b_content_types_are_in_priority_order():
    s = C.system_prompt_b(RUB)
    positions = [s.index(f"- {c}:") for c in R.CONTENT_TYPE_IDS]
    assert positions == sorted(positions)
    assert "FIRST rule that applies" in s


def test_schema_b_shape():
    sch = C.schema_b(RUB)
    props = sch["properties"]
    assert sch["required"] == list(props) and sch["additionalProperties"] is False
    assert props["basis"]["enum"] == list(R.BASIS_IDS)
    assert props["content_type"]["enum"] == list(R.CONTENT_TYPE_IDS)
    assert props["difficulty"]["enum"] == list(R.DIFFICULTY_IDS)
    assert set(props["alignment"]["properties"]) == set(R.ALIGNMENT_IDS)
    assert all(v == {"type": "integer", "minimum": 0, "maximum": 10}
               for v in props["alignment"]["properties"].values())
    assert props["cover_minutes"]["maximum"] == 20160 and props["understand_minutes"]["maximum"] == 43200
    assert props["stated_unit"]["enum"] == list(R.STATED_UNIT_IDS)
    assert props["stated_count"] == {"type": "number", "minimum": 0}


def test_schema_b_field_order_puts_reasoning_before_answers():
    keys = list(C.schema_b(RUB)["properties"])
    assert keys.index("stated_size") < keys.index("stated_count") < keys.index("stated_unit") \
        < keys.index("basis") < keys.index("cover_minutes")
    assert keys.index("key_points") < keys.index("content_type")
    assert keys.index("alignment") < keys.index("summary") < keys.index("self_confidence")


def test_user_prompt_shows_post_and_optional_parts():
    plain = C.user_prompt_a(I.Post(1, "hello there"))
    assert "<<<\nhello there\n>>>" in plain
    assert "LINK PREVIEW" not in plain and "LINKS IN POST" not in plain and "cut" not in plain
    rich = C.user_prompt_a(I.Post(1, "hi", attachment_title="Title", attachment_site="site.com",
                                  links="https://x.y", truncated=True))
    assert "LINK PREVIEW: Title (site.com)" in rich and "LINKS IN POST: https://x.y" in rich
    assert "has been cut" in rich


def test_user_prompt_b_adds_baseline_and_word_count():
    u = C.user_prompt_b(I.Post(1, "one two three"), 7)
    assert "BASELINE (reading time of the post itself): 7 minutes" in u and "WORDS IN POST: 3" in u


# ============================================================ checking answers

def test_check_a_valid(tax):
    assert C.check_a(GOOD_A, tax) == ([], [])
    assert C.check_a({"evidence": "x", "subtopic_id": "other.general"}, tax) == ([], [])


@pytest.mark.parametrize("bad", [
    "text", None, {"evidence": "x"}, {"evidence": "x", "subtopic_id": "rag.nonsense"},
    {"evidence": 5, "subtopic_id": "rag.chunking"},
])
def test_check_a_hard_problems(tax, bad):
    hard, soft = C.check_a(bad, tax)
    assert hard and soft == []


def test_check_b_valid():
    assert C.check_b(good_b(), RUB) == ([], [])


@pytest.mark.parametrize("change", [
    {"basis": "guess"}, {"content_type": "blog"}, {"difficulty": "easy"},
    {"key_points": []}, {"key_points": ["a"] * 6}, {"key_points": "text"}, {"key_points": [1]},
    {"cover_minutes": "5"}, {"understand_minutes": None}, {"self_confidence": True},
    {"summary": 5}, {"stated_size": None}, {"time_reason": 1},
    {"stated_count": "3"}, {"stated_count": None}, {"stated_count": True}, {"stated_unit": "years"},
    {"stated_unit": None},
    {"alignment": {"applicability": 5}}, {"alignment": "x"},
    {"alignment": {**good_b()["alignment"], "extra": 1}},
    {"alignment": {**good_b()["alignment"], "applicability": "8"}},
])
def test_check_b_hard_problems(change):
    hard, _ = C.check_b(good_b(**change), RUB)
    assert hard


def test_check_b_not_an_object():
    assert C.check_b([1], RUB)[0]


def test_check_b_missing_key_is_hard():
    b = good_b()
    del b["summary"]
    assert C.check_b(b, RUB)[0]


@pytest.mark.parametrize("change,fragment", [
    ({"cover_minutes": 0}, "cover_minutes must be at least 1"),
    ({"understand_minutes": -3}, "understand_minutes must be at least 1"),
    ({"self_confidence": 1.5}, "self_confidence"),
    ({"summary": words(30)}, "summary has 30 words"),
    ({"summary": words(300)}, "summary has 300 words"),
    ({"summary": words(30) + "\n" + words(30)}, "no line breaks"),
    ({"summary": "- " + words(60)}, "bullet"),
    ({"summary": "1. " + words(60)}, "bullet"),
    ({"basis": "stated_in_post"}, "stated_size is empty"),
    ({"stated_size": "8-week course"}, "basis is not stated_in_post"),
    ({"stated_count": -2}, "stated_count must be 0 or more"),
])
def test_check_b_soft_problems(change, fragment):
    hard, soft = C.check_b(good_b(**change), RUB)
    assert hard == [] and any(fragment in s for s in soft)


@pytest.mark.parametrize("value", [-1, 11, 7.5])
def test_check_b_alignment_subscore_range(value):
    al = {**good_b()["alignment"], "currency": value}
    hard, soft = C.check_b(good_b(alignment=al), RUB)
    assert hard == [] and any("alignment.currency" in s for s in soft)


def test_check_b_summary_limits_are_inclusive():
    for n in (40, 220):
        assert C.check_b(good_b(summary=words(n)), RUB) == ([], [])
    assert C.check_b(good_b(summary=words(39)), RUB)[1]
    assert C.check_b(good_b(summary=words(221)), RUB)[1]


def test_check_b_stated_size_with_matching_basis_is_fine():
    b = good_b(basis="stated_in_post", stated_size="8-week course")
    assert C.check_b(b, RUB) == ([], [])


# ============================================================ repairs

def test_repair_valid_answer_changes_nothing():
    fixed, notes = C.repair_b(good_b(), RUB)
    assert fixed == good_b() and notes == []


def test_repair_does_not_change_the_input():
    original = good_b(cover_minutes=0)
    snapshot = copy.deepcopy(original)
    C.repair_b(original, RUB)
    assert original == snapshot


def test_repair_minutes_and_ranges():
    b = good_b(cover_minutes=0, understand_minutes=-4, self_confidence=1.7,
               alignment={**good_b()["alignment"], "applicability": 12, "currency": -2, "hands_on": 7.4})
    fixed, notes = C.repair_b(b, RUB)
    assert fixed["cover_minutes"] == 1 and fixed["understand_minutes"] == 1
    assert fixed["self_confidence"] == 1.0
    assert fixed["alignment"]["applicability"] == 10 and fixed["alignment"]["currency"] == 0
    assert fixed["alignment"]["hands_on"] == 7
    assert {"repaired:cover_minutes", "repaired:understand_minutes", "repaired:self_confidence",
            "repaired:alignment.applicability", "repaired:alignment.currency",
            "repaired:alignment.hands_on"} <= set(notes)


def test_repair_float_minutes_become_whole_numbers():
    fixed, _ = C.repair_b(good_b(cover_minutes=5.6), RUB)
    assert fixed["cover_minutes"] == 6 and isinstance(fixed["cover_minutes"], int)


def test_repair_unsupported_stated_basis_falls_back_to_post_only():
    fixed, notes = C.repair_b(good_b(basis="stated_in_post", stated_size=""), RUB)
    assert fixed["basis"] == "post_only" and "repaired:basis" in notes


def test_repair_unused_stated_size_is_dropped():
    fixed, notes = C.repair_b(good_b(basis="estimated_from_post", stated_size="8 weeks"), RUB)
    assert fixed["stated_size"] == "" and fixed["basis"] == "estimated_from_post"
    assert "repaired:stated_size" in notes


def test_repair_negative_stated_count_becomes_zero():
    # with the post's words present, only this rule (not the "no words" rule) can fix the number
    fixed, notes = C.repair_b(good_b(basis="stated_in_post", stated_size="5 hours", stated_count=-5,
                                     stated_unit="hours"), RUB)
    assert fixed["stated_count"] == 0 and notes == ["repaired:stated_count", "repaired:basis_unconvertible"]


def test_repair_number_without_the_posts_words_is_dropped():
    fixed, notes = C.repair_b(good_b(stated_count=8, stated_unit="weeks"), RUB)
    assert (fixed["stated_count"], fixed["stated_unit"]) == (0, "none")
    assert notes == ["repaired:stated_count"]


def test_repair_dropped_stated_size_takes_its_number_with_it():
    fixed, notes = C.repair_b(good_b(basis="estimated_from_post", stated_size="8 weeks",
                                     stated_count=8, stated_unit="weeks"), RUB)
    assert (fixed["stated_size"], fixed["stated_count"], fixed["stated_unit"]) == ("", 0, "none")
    assert "repaired:stated_size" in notes


def test_repair_usable_stated_size_changes_nothing():
    b = good_b(basis="stated_in_post", stated_size="20 interview questions", stated_count=20,
               stated_unit="questions")
    fixed, notes = C.repair_b(b, RUB)
    assert fixed == b and notes == []


@pytest.mark.parametrize("size,count,unit", [
    ("1:1 mentorship, unlimited mock interviews", 0, "none"),     # a list of features is not a size
    ("22 courses", 22, "other"),                                    # a size, but no duration
    ("8-week course", 0, "weeks"),                                  # a unit without a number
])
def test_repair_unconvertible_stated_size_falls_back_to_post_only(size, count, unit):
    fixed, notes = C.repair_b(good_b(basis="stated_in_post", stated_size=size, stated_count=count,
                                     stated_unit=unit), RUB)
    assert fixed["basis"] == "post_only" and notes == ["repaired:basis_unconvertible"]
    assert fixed["stated_size"] == size and fixed["stated_count"] == count      # kept for the record


def test_repair_count_that_is_not_in_the_words_falls_back_to_post_only():
    b = good_b(basis="stated_in_post", stated_size="20 interview questions", stated_count=200,
               stated_unit="questions")
    fixed, notes = C.repair_b(b, RUB)
    assert fixed["basis"] == "post_only" and notes == ["repaired:basis_count_mismatch"]


def test_repair_count_may_be_any_of_the_numbers_in_the_words():
    b = good_b(basis="stated_in_post", stated_size="30 questions in 5 hours", stated_count=5, stated_unit="hours")
    assert C.repair_b(b, RUB)[1] == []


def test_repair_number_words_cannot_be_compared_and_are_accepted():
    b = good_b(basis="stated_in_post", stated_size="an eight-week course", stated_count=8, stated_unit="weeks")
    assert C.repair_b(b, RUB)[1] == []


def test_repair_decimal_counts_with_a_comma_or_a_point():
    for words_ in ("1.5 hours of video", "1,5 hours of video"):
        b = good_b(basis="stated_in_post", stated_size=words_, stated_count=1.5, stated_unit="hours")
        assert C.repair_b(b, RUB)[1] == []


def test_repair_summary_bullets_and_line_breaks():
    messy = "- First point here.\n- Second point there.\n\n• Third point."
    fixed, notes = C.repair_b(good_b(summary=messy), RUB)
    assert fixed["summary"] == "First point here. Second point there. Third point."
    assert "repaired:summary_format" in notes


def test_repair_long_summary_is_cut_at_a_sentence_end():
    fixed, notes = C.repair_b(good_b(summary=words(300)), RUB)
    n = len(fixed["summary"].split())
    assert n <= 220 and fixed["summary"].endswith(".") and n % 8 == 0     # sentences are 8 words long
    assert n == 216 and "repaired:summary_trimmed" in notes


def test_repair_long_summary_without_sentence_ends_is_cut_at_the_limit():
    fixed, _ = C.repair_b(good_b(summary=" ".join(["word"] * 300)), RUB)
    assert len(fixed["summary"].split()) == 220


# ============================================================ coherence warnings and confidence

def warn(**kw):
    args = dict(content_type="tutorial", difficulty="intermediate", time={"understand_minutes": 60},
                stated_size="", post_text="plain post", summary=words(60), rub=RUB)
    args.update(kw)
    return C.coherence_warnings(**args)


def test_no_warnings_for_a_clean_answer():
    assert warn() == []


def test_full_project_needs_a_long_understand_time():
    assert "full_project_short_understand" in warn(content_type="full_project", time={"understand_minutes": 179})
    assert "full_project_short_understand" not in warn(content_type="full_project", time={"understand_minutes": 180})


@pytest.mark.parametrize("difficulty,expected", [("beginner", False), ("intermediate", False),
                                                 ("advanced", True), ("expert", True)])
def test_news_should_not_be_hard(difficulty, expected):
    assert ("news_hard" in warn(content_type="news", difficulty=difficulty)) is expected


def test_stated_size_numbers_must_appear_in_the_post():
    assert "stated_size_not_in_post" in warn(stated_size="12-week course", post_text="an 8 week course")
    assert "stated_size_not_in_post" not in warn(stated_size="8-week course", post_text="an 8 week course")
    assert "stated_size_not_in_post" not in warn(stated_size="eight weeks", post_text="a long course")  # no digits to check


def test_every_number_of_a_stated_size_is_checked():
    # "8" is in the post but "12" is not: one invented number is enough to flag it
    assert "stated_size_not_in_post" in warn(stated_size="8 weeks, 12 hours", post_text="an 8 week course")
    assert "stated_size_not_in_post" not in warn(stated_size="8 weeks, 5 hours", post_text="8 weeks, 5 hours a week")


def test_summary_warnings():
    assert "summary_short" in warn(summary=words(10))
    assert "summary_missing" in warn(summary="")
    assert "summary_short" not in warn(summary=words(40))


def test_agreement_levels(tax):
    assert C.agreement("rag.chunking", "rag.chunking", tax) == 1.0
    assert C.agreement("rag.chunking", "rag.evaluation", tax) == 0.7          # same topic
    assert C.agreement("rag.chunking", "careers.interviews", tax) == 0.4      # different topic
    assert C.agreement("other.general", "rag.chunking", tax) == 0.4
    assert C.agreement("rag.chunking", "other.general", tax) == 0.4
    assert C.agreement("other.general", "other.general", tax) == 1.0
    assert C.agreement("rag.chunking", None, tax) is None


def conf(**kw):
    args = dict(self_confidence=0.6, agree=1.0, first_try=True, partial=False, penalties=0)
    args.update(kw)
    return C.compute_confidence(**args)[0]


def test_confidence_worked_examples():
    # 0.45 x 0.6 + 0.35 x 1.0 + 0.20 x 1.0 = 0.27 + 0.35 + 0.20 = 0.82
    assert conf() == 0.82
    # agreement 0.4: 0.27 + 0.14 + 0.20 = 0.61
    assert conf(agree=0.4) == 0.61
    # after a retry the validity part is 0.7: 0.27 + 0.35 + 0.14 = 0.76
    assert conf(first_try=False) == 0.76
    # partial text: 0.82 x 0.7 = 0.574
    assert conf(partial=True) == 0.57
    # two warnings: 0.82 - 0.2
    assert conf(penalties=2) == 0.62
    # no clustering opinion: (0.45 x 0.8 + 0.20 x 1.0) / 0.65 = 0.8615...
    assert conf(self_confidence=0.8, agree=None) == 0.86


def test_confidence_is_clamped():
    assert conf(penalties=20) == 0.0
    assert conf(self_confidence=1.0) <= 1.0


def test_confidence_parts_are_reported():
    _, parts = C.compute_confidence(self_confidence=0.6, agree=0.7, first_try=False, partial=True, penalties=1)
    assert parts == {"self": 0.6, "agree": 0.7, "valid": 0.7, "partial": True, "penalties": 1}


def test_needs_review_threshold_and_other():
    assert C.needs_review(0.59, "rag.chunking") is True
    assert C.needs_review(0.60, "rag.chunking") is False
    assert C.needs_review(0.99, "other.general") is True


# ============================================================ context and cache key

def test_cache_key_is_stable(tax):
    assert C.cache_key(POST, make_ctx(tax)) == C.cache_key(POST, make_ctx(tax))


def test_cache_key_changes_with_every_ingredient(tax, monkeypatch):
    base = C.cache_key(POST, make_ctx(tax))
    other_text = I.Post(1, POST.text + " more")
    assert C.cache_key(other_text, make_ctx(tax)) != base
    assert C.cache_key(POST, make_ctx(tax, model="m2")) != base
    assert C.cache_key(POST, make_ctx(tax, think=True)) != base
    assert C.cache_key(POST, make_ctx(tax, num_ctx=4096)) != base
    assert C.cache_key(POST, make_ctx(tax, options={"presence_penalty": 0})) != base
    new_tax = I.Taxonomy(version=tax.version, subtopics=tax.subtopics, sha="other", path=tax.path, by_id=tax.by_id)
    assert C.cache_key(POST, make_ctx(new_tax)) != base
    new_rub = R.Rubrics(data=RUB.data, sha="other", path=RUB.path)
    assert C.cache_key(POST, make_ctx(tax, rubrics=new_rub)) != base
    monkeypatch.setattr(C, "PROMPT_VERSION", "classify-test")
    assert C.cache_key(POST, make_ctx(tax)) != base


def test_reposts_get_their_own_cache_key(tax):
    # identical text, different posts: both must be classified and stored
    ctx = make_ctx(tax)
    assert C.cache_key(I.Post(1, "same text"), ctx) != C.cache_key(I.Post(2, "same text"), ctx)


def test_prompt_sha_follows_prompt_text(tax):
    base = make_ctx(tax).prompt_sha
    data = copy.deepcopy(RUB.data)
    data["role_profile"] = "something else"
    changed = make_ctx(tax, rubrics=R.Rubrics(data=data, sha=RUB.sha, path=RUB.path))
    assert changed.prompt_sha != base


def test_prompt_sha_follows_the_user_prompt_template(tax, monkeypatch):
    base = make_ctx(tax).prompt_sha
    original = C.user_prompt_a
    monkeypatch.setattr(C, "user_prompt_a", lambda post: original(post) + "extra line")
    assert make_ctx(tax).prompt_sha != base


# ============================================================ classify_post

def run(tax, chat=None, post=POST, **ctx_kw):
    chat = chat or FakeChat()
    ctx = make_ctx(tax, **ctx_kw)
    return C.classify_post(post, ctx, chat=chat), chat, ctx


def test_happy_path_record(tax):
    rec, chat, ctx = run(tax)
    assert rec["activity_id"] == 1 and rec["subtopic_id"] == "rag.chunking" and rec["topic_id"] == "rag"
    assert rec["difficulty"] == "intermediate" and rec["content_type"] == "tutorial"
    # post has 10 words -> baseline 1; post_only clamp 0.5..3 -> model's 2 stays -> shown 5;
    # understand: 2 x 2.5 x 1.2 = 6, geometric mean with 10 = 7.75 -> shown 10
    t = rec["time"]
    assert (t["cover_minutes"], t["understand_minutes"]) == (5, 10)
    assert (t["cover_bucket"], t["understand_bucket"]) == ("<15m", "<15m")
    assert t["basis"] == "post_only" and t["baseline_cover_minutes"] == 1 and t["stated_size"] == ""
    assert (t["stated_count"], t["stated_unit"], t["stated_minutes"]) == (0, "none", None)
    assert (t["model_cover_minutes"], t["model_understand_minutes"]) == (2, 10)
    assert rec["alignment_score"] == 7.0 and rec["alignment_band"] == "Highly relevant"
    assert rec["summary"] == words(60) and rec["key_points"] == ["explains chunking"]
    assert rec["rationales"] == {"topic": "chunking documents", "content_type": "Teaches step by step.",
                                 "difficulty": "Needs Python.", "time": "Short post.", "alignment": "Useful."}
    # 0.45 x 0.8 + 0.35 x 1.0 (same subtopic as clustering) + 0.20 = 0.91
    assert rec["confidence"] == 0.91 and rec["needs_review"] is False and rec["warnings"] == []
    assert rec["clustering_subtopic_id"] == "rag.chunking"
    assert rec["attempts"] == {"topic": 1, "rest": 1} and rec["retry_reasons"] == {"topic": [], "rest": []}
    assert rec["cache_key"] == C.cache_key(POST, ctx)
    assert rec["model"] == "m1" and rec["prompt_version"] == C.PROMPT_VERSION
    assert rec["taxonomy_version"] == "v1" and rec["rubrics_version"] == "v1"
    assert rec["rubrics_sha"] == RUB.sha and rec["prompt_sha"] == ctx.prompt_sha


def test_record_is_json_serialisable(tax):
    rec, _, _ = run(tax)
    assert json.loads(json.dumps(rec))["activity_id"] == 1


def test_usage_is_summed_over_both_calls(tax):
    rec, _, _ = run(tax)
    assert rec["input_tokens"] == 300 and rec["output_tokens"] == 170 and rec["seconds"] == 5.5


def test_two_calls_with_the_right_arguments(tax):
    rec, chat, ctx = run(tax, num_ctx=4096, think=True, options={"presence_penalty": 0}, url="http://x:1")
    assert [c["kind"] for c in chat.calls] == ["a", "b"]
    a, b = chat.calls
    assert a["system"] == ctx.system_a and b["system"] == ctx.system_b
    assert a["schema"] == ctx.schema_a and b["schema"] == ctx.schema_b
    assert a["user"] == C.user_prompt_a(POST) and b["user"] == C.user_prompt_b(POST, 1)
    for call in chat.calls:
        assert call["model"] == "m1" and call["num_ctx"] == 4096 and call["think"] is True
        assert call["url"] == "http://x:1" and call["options"] == {"presence_penalty": 0}
    assert a["num_predict"] == 400 and b["num_predict"] == 1500


def test_other_general_needs_review(tax):
    chat = FakeChat(a=[{"evidence": "nothing fits", "subtopic_id": "other.general"}])
    rec, _, _ = run(tax, chat, assignments={1: "other.general"})
    assert rec["topic_id"] == "other" and rec["needs_review"] is True


def test_disagreement_with_clustering_lowers_confidence(tax):
    same, _, _ = run(tax)
    diff, _, _ = run(tax, assignments={1: "careers.interviews"})
    assert same["confidence"] == 0.91
    assert diff["confidence"] == 0.70            # 0.45 x 0.8 = 0.36; 0.35 x 0.4 = 0.14; plus 0.20


def test_missing_clustering_opinion(tax):
    rec, _, _ = run(tax, assignments={})
    assert "no_clustering_opinion" in rec["warnings"] and rec["clustering_subtopic_id"] is None
    assert rec["confidence"] == 0.86             # (0.36 + 0.20) / 0.65


def test_cut_off_post_is_partial(tax):
    post = I.Post(1, "A post that was not fully expanded …see more")
    rec, _, _ = run(tax, post=post)
    assert "text_partial" in rec["warnings"]
    assert rec["confidence"] == 0.64             # 0.91 x 0.7 = 0.637


def test_truncated_post_is_partial(tax):
    rec, _, _ = run(tax, post=I.Post(1, "long post", truncated=True))
    assert "text_partial" in rec["warnings"]


def test_advertised_course_keeps_its_long_duration(tax):
    post = I.Post(1, "Join my 8 week course on agents, 5 hours a week.")
    b = good_b(basis="stated_in_post", stated_size="8 week course, 5 hours a week",
               stated_count=8, stated_unit="weeks", cover_minutes=2400, understand_minutes=4800)
    rec, _, _ = run(tax, FakeChat(b=[b]), post=post)
    t = rec["time"]
    assert t["cover_minutes"] == 2400 and t["cover_bucket"] == ">3d" and t["basis"] == "stated_in_post"
    assert t["stated_size"] == "8 week course, 5 hours a week"
    assert (t["stated_count"], t["stated_unit"], t["stated_minutes"]) == (8, "weeks", 2400)
    assert "stated_size_not_in_post" not in rec["warnings"]


def test_invented_duration_is_flagged(tax):
    post = I.Post(1, "Join my new course on agents.")
    b = good_b(basis="stated_in_post", stated_size="12 week course", stated_count=12, stated_unit="weeks",
               cover_minutes=3000, understand_minutes=6000)
    rec, _, _ = run(tax, FakeChat(b=[b]), post=post)
    assert "stated_size_not_in_post" in rec["warnings"]
    assert rec["confidence"] == 0.81             # 0.91 - 0.10


def stated_b(size, count, unit, **over):
    return good_b(basis="stated_in_post", stated_size=size, stated_count=count, stated_unit=unit, **over)


def test_twenty_questions_are_sixty_minutes_whatever_the_model_says(tax):
    # real case from the first run: the model wrote "3 minutes each" but answered 15 minutes
    post = I.Post(1, "Here are 20 important LLM interview questions with answers.")
    b = stated_b("20 important LLM interview questions", 20, "questions",
                 content_type="interview_question_set", cover_minutes=15, understand_minutes=45)
    rec, _, _ = run(tax, FakeChat(b=[b]), post=post)
    t = rec["time"]
    assert (t["cover_minutes"], t["cover_bucket"]) == (60, "1-3h")
    assert t["understand_minutes"] == 120            # sqrt(45 x (60 x 2.5 x 2.0)) = 116.2 -> 120
    assert (t["stated_minutes"], t["model_cover_minutes"]) == (60, 15) and t["basis"] == "stated_in_post"
    assert rec["warnings"] == []


def test_a_list_of_features_is_not_a_size(tax):
    # real case: "1:1 mentorship, unlimited mock interviews..." was taken for a stated size
    post = I.Post(1, "My AI PM course: 1:1 mentorship, unlimited mock interviews and a job board.")
    b = stated_b("1:1 mentorship, unlimited mock interviews", 0, "none", cover_minutes=5, understand_minutes=10)
    rec, _, _ = run(tax, FakeChat(b=[b]), post=post)
    t = rec["time"]
    assert t["basis"] == "post_only" and t["stated_minutes"] is None
    assert t["stated_size"] == "1:1 mentorship, unlimited mock interviews"       # kept for the record
    assert "repaired:basis_unconvertible" in rec["warnings"]
    assert rec["confidence"] == 0.85                  # a repair counts like a retry: valid part 0.7


def test_a_count_of_courses_has_no_duration(tax):
    # real case: "22 courses" must not turn into an invented number of minutes
    post = I.Post(1, "The bundle has 22 courses for interview preparation.")
    rec, _, _ = run(tax, FakeChat(b=[stated_b("22 courses", 22, "other")]), post=post)
    t = rec["time"]
    assert t["basis"] == "post_only" and t["stated_minutes"] is None
    assert (t["stated_count"], t["stated_unit"]) == (22, "other")
    assert "repaired:basis_unconvertible" in rec["warnings"]


def test_a_count_that_is_not_in_the_posts_words_is_not_trusted(tax):
    post = I.Post(1, "Here are 20 interview questions.")
    rec, _, _ = run(tax, FakeChat(b=[stated_b("20 interview questions", 200, "questions")]), post=post)
    assert rec["time"]["basis"] == "post_only" and "repaired:basis_count_mismatch" in rec["warnings"]


def test_retry_reasons_are_recorded(tax):
    chat = FakeChat(a=[{"evidence": "x", "subtopic_id": "rag.nonsense"}, GOOD_A],
                    b=[good_b(summary=words(10)), good_b()])
    rec, _, _ = run(tax, chat)
    why = rec["retry_reasons"]
    assert len(why["topic"]) == 1 and "rag.nonsense" in why["topic"][0]
    assert len(why["rest"]) == 1 and "summary has 10 words" in why["rest"][0]


def test_retry_reason_for_an_unusable_reply(tax):
    rec, _, _ = run(tax, FakeChat(b=[LLMError("Answer was not valid JSON"), good_b()]))
    assert rec["retry_reasons"] == {"topic": [], "rest": ["Answer was not valid JSON"]}


def test_retry_reason_is_kept_even_when_the_retry_breaks(tax):
    chat = FakeChat(b=[good_b(summary=words(10)), LLMError("Answer was not valid JSON")])
    rec, _, _ = run(tax, chat)
    assert "summary has 10 words" in rec["retry_reasons"]["rest"][0]


def test_retry_after_unknown_subtopic(tax):
    chat = FakeChat(a=[{"evidence": "x", "subtopic_id": "rag.nonsense"}, GOOD_A])
    rec, chat, _ = run(tax, chat)
    assert [c["kind"] for c in chat.calls] == ["a", "a", "b"]
    retry_user = chat.calls[1]["user"]
    assert retry_user.startswith(C.user_prompt_a(POST)) and "REJECTED" in retry_user and "rag.nonsense" in retry_user
    assert rec["attempts"] == {"topic": 2, "rest": 1}
    assert rec["confidence"] == 0.85             # valid part 0.7: 0.36 + 0.35 + 0.14


def test_retry_after_an_unusable_reply(tax):
    chat = FakeChat(b=[LLMError("Answer was not valid JSON"), good_b()])
    rec, chat, _ = run(tax, chat)
    assert [c["kind"] for c in chat.calls] == ["a", "b", "b"]
    assert "not valid JSON" in chat.calls[2]["user"] and rec["attempts"]["rest"] == 2


def test_retry_after_a_soft_problem(tax):
    bad = good_b(summary=words(10))
    rec, chat, _ = run(tax, FakeChat(b=[bad, good_b()]))
    assert rec["attempts"]["rest"] == 2 and rec["summary"] == words(60) and rec["warnings"] == []


def test_two_failures_raise(tax):
    chat = FakeChat(a=[LLMError("bad json"), LLMError("bad json again")])
    with pytest.raises(C.ClassifyFailed, match="bad json again"):
        run(tax, chat)
    assert [c["kind"] for c in chat.calls] == ["a", "a"]        # B is never asked


def test_hard_problem_twice_raises(tax):
    bad = {"evidence": "x", "subtopic_id": "nope"}
    with pytest.raises(C.ClassifyFailed):
        run(tax, FakeChat(a=[bad, bad]))


def test_ollama_unavailable_is_not_retried(tax):
    chat = FakeChat(a=[OllamaUnavailable("not running")])
    with pytest.raises(OllamaUnavailable):
        run(tax, chat)
    assert len(chat.calls) == 1


def test_soft_problems_that_persist_are_repaired_and_marked(tax):
    bad = good_b(alignment={**good_b()["alignment"], "applicability": 12})
    rec, chat, _ = run(tax, FakeChat(b=[bad, bad]))
    assert rec["alignment_subscores"]["applicability"] == 10
    assert "repaired:alignment.applicability" in rec["warnings"]
    assert rec["attempts"]["rest"] == 2
    # 12 -> 10: 0.3x10 + 1.2 + 1.8 + 0.7 + 0.5 + 0.4 = 7.6
    assert rec["alignment_score"] == 7.6
    assert rec["confidence"] == 0.85


def test_first_answer_is_kept_when_only_the_retry_breaks(tax):
    soft_bad = good_b(summary=words(10))
    chat = FakeChat(b=[soft_bad, LLMError("Answer was not valid JSON")])
    rec, _, _ = run(tax, chat)
    assert rec["attempts"]["rest"] == 2 and "summary_short" in rec["warnings"]


def test_summary_that_stays_short_lowers_confidence(tax):
    short = good_b(summary=words(10))
    rec, _, _ = run(tax, FakeChat(b=[short, short]))
    assert "summary_short" in rec["warnings"]
    assert rec["confidence"] == 0.75             # 0.36 + 0.35 + 0.14 = 0.85 (retry), minus 0.10
