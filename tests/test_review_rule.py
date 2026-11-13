"""Tests for pipeline/review_rule.py (Step 13): the review rule file and the decision."""
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import classify_inputs as I  # noqa: E402
import review_rule as RV  # noqa: E402

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "Retrieval-Augmented Generation", "description": "RAG systems.",
         "subtopics": [{"id": "rag.chunking", "name": "Chunking", "description": "Splitting documents."},
                       {"id": "rag.evaluation", "name": "RAG Evaluation", "description": "Measuring RAG."}]},
        {"id": "careers", "name": "Career Development", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
    ],
    "other": {"id": "other.general", "name": "Other / General", "description": "Fits nothing."},
}

GOOD = {"version": "v1", "confidence_below": 0.78, "other_general": True,
        "topic_disagrees_with_clustering": True}


@pytest.fixture(scope="module")
def tax(tmp_path_factory):
    p = tmp_path_factory.mktemp("tax") / "tax.yaml"
    p.write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    return I.load_taxonomy(p)


def write(path: Path, data) -> Path:
    path.write_bytes(yaml.safe_dump(data).encode("utf-8"))
    return path


def rule(**over):
    base = dict(version="t", confidence_below=0.78, other_general=True, topic_disagrees=True, sha="s")
    base.update(over)
    return RV.ReviewRule(**base)


# ============================================================ the shipped file

def test_the_shipped_file_is_valid_and_has_the_calibrated_values():
    r = RV.load_review_rule()
    assert r.version == "v1" and r.confidence_below == 0.78
    assert r.other_general is True and r.topic_disagrees is True
    assert len(r.sha) == 64 and set(r.sha) <= set("0123456789abcdef")


def test_the_default_path_is_the_rubrics_folder():
    assert RV.DEFAULT_PATH == ROOT / "rubrics" / "review_rule.v1.yaml" and RV.DEFAULT_PATH.exists()


def test_reasons_have_fixed_names_in_a_fixed_order():
    assert RV.REASONS == ("low_confidence", "other_general", "topic_disagrees_with_clustering")


# ============================================================ validation

def test_a_good_file_has_no_problems():
    assert RV.validate_review_rule(GOOD) == []


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 0.5])
def test_confidence_below_accepts_numbers_from_0_to_1(value):
    assert RV.validate_review_rule({**GOOD, "confidence_below": value}) == []


@pytest.mark.parametrize("value", [-0.01, 1.01, 5, "0.8", None, True, False, math.nan, math.inf, [0.5]])
def test_confidence_below_rejects_everything_else(value):
    problems = RV.validate_review_rule({**GOOD, "confidence_below": value})
    assert len(problems) == 1 and "confidence_below must be a number from 0 to 1" in problems[0]


@pytest.mark.parametrize("key", ["other_general", "topic_disagrees_with_clustering"])
@pytest.mark.parametrize("value", ["yes", 1, 0, None, "true"])
def test_the_switches_must_be_true_or_false(key, value):
    problems = RV.validate_review_rule({**GOOD, key: value})
    assert len(problems) == 1 and f"{key} must be true or false" in problems[0]


@pytest.mark.parametrize("value", ["", "  ", None, 1])
def test_version_must_be_text(value):
    problems = RV.validate_review_rule({**GOOD, "version": value})
    assert problems == ["version must be a non-empty text"]


def test_missing_keys_are_named():
    problems = RV.validate_review_rule({"version": "v1"})
    assert problems == ["missing keys: ['confidence_below', 'other_general', 'topic_disagrees_with_clustering']"]


def test_a_typo_in_a_key_is_not_silently_ignored():
    data = {k: v for k, v in GOOD.items() if k != "confidence_below"}
    data["confidence_belov"] = 0.9
    problems = RV.validate_review_rule(data)
    assert any("missing keys: ['confidence_below']" in p for p in problems)
    assert any("unknown keys: ['confidence_belov']" in p for p in problems)


@pytest.mark.parametrize("data", [None, [], "text", 5])
def test_the_top_level_must_be_a_mapping(data):
    assert RV.validate_review_rule(data) == ["the file must contain a mapping at the top level"]


# ============================================================ reading the file

def test_load_a_custom_file(tmp_path):
    p = write(tmp_path / "r.yaml", {**GOOD, "version": "v9", "confidence_below": 1, "other_general": False})
    r = RV.load_review_rule(p)
    assert (r.version, r.confidence_below, r.other_general, r.topic_disagrees) == ("v9", 1.0, False, True)
    assert isinstance(r.confidence_below, float)           # 1 in the file becomes 1.0


def test_load_lists_every_problem_together(tmp_path):
    p = write(tmp_path / "r.yaml", {**GOOD, "confidence_below": 7, "other_general": "yes"})
    with pytest.raises(RV.ReviewRuleError) as exc:
        RV.load_review_rule(p)
    msg = str(exc.value)
    assert "confidence_below must be a number" in msg and "other_general must be true or false" in msg


def test_unreadable_file(tmp_path):
    data, sha, problems = RV.check_file(tmp_path / "missing.yaml")
    assert data is None and sha is None and problems[0].startswith("cannot read")
    with pytest.raises(RV.ReviewRuleError, match="cannot read"):
        RV.load_review_rule(tmp_path / "missing.yaml")


def test_file_that_is_not_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_bytes(b"version: [unclosed\n")
    _, _, problems = RV.check_file(p)
    assert problems and "is not valid YAML" in problems[0]


def test_file_that_is_not_utf8(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_bytes(b"version: \xff\xfe\n")
    _, _, problems = RV.check_file(p)
    assert problems and "is not valid YAML" in problems[0]


def test_the_fingerprint_ignores_windows_line_endings(tmp_path):
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    text = "version: v1\nconfidence_below: 0.78\nother_general: true\ntopic_disagrees_with_clustering: true\n"
    lf.write_bytes(text.encode("utf-8"))
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert RV.load_review_rule(lf).sha == RV.load_review_rule(crlf).sha


def test_the_fingerprint_changes_when_the_number_changes(tmp_path):
    a = RV.load_review_rule(write(tmp_path / "a.yaml", GOOD))
    b = RV.load_review_rule(write(tmp_path / "b.yaml", {**GOOD, "confidence_below": 0.8}))
    assert a.sha != b.sha


# ============================================================ the decision

def reasons(r, tax, confidence=0.95, subtopic="rag.chunking", clustering="rag.chunking"):
    return RV.review_reasons(r, confidence=confidence, subtopic_id=subtopic,
                             clustering_subtopic_id=clustering, taxonomy=tax)


def test_a_confident_agreeing_post_is_not_reviewed(tax):
    assert reasons(rule(), tax) == []


def test_confidence_must_be_strictly_below_the_line(tax):
    assert reasons(rule(), tax, confidence=0.77) == ["low_confidence"]
    assert reasons(rule(), tax, confidence=0.78) == []
    assert reasons(rule(), tax, confidence=0.79) == []


def test_a_line_of_zero_switches_the_confidence_signal_off(tax):
    assert reasons(rule(confidence_below=0.0), tax, confidence=0.0) == []


def test_a_line_of_one_flags_everything_below_one(tax):
    r = rule(confidence_below=1.0)
    assert reasons(r, tax, confidence=0.99) == ["low_confidence"]
    assert reasons(r, tax, confidence=1.0) == []


def test_other_general_flag_and_its_switch(tax):
    kw = dict(subtopic="other.general", clustering="other.general")
    assert reasons(rule(), tax, **kw) == ["other_general"]
    assert reasons(rule(other_general=False), tax, **kw) == []


def test_a_topic_disagreement_flag_and_its_switch(tax):
    kw = dict(subtopic="rag.chunking", clustering="careers.interviews")
    assert reasons(rule(), tax, **kw) == ["topic_disagrees_with_clustering"]
    assert reasons(rule(topic_disagrees=False), tax, **kw) == []


def test_the_same_topic_with_a_different_subtopic_is_fine(tax):
    assert reasons(rule(), tax, subtopic="rag.chunking", clustering="rag.evaluation") == []


@pytest.mark.parametrize("clustering", [None, "other.general"])
def test_no_clustering_opinion_is_never_a_disagreement(tax, clustering):
    assert reasons(rule(), tax, subtopic="rag.chunking", clustering=clustering) == []


def test_the_model_saying_other_against_a_real_clustering_topic_gives_two_reasons(tax):
    assert reasons(rule(), tax, subtopic="other.general", clustering="rag.chunking") == \
        ["other_general", "topic_disagrees_with_clustering"]


def test_reasons_come_in_the_fixed_order(tax):
    assert reasons(rule(), tax, confidence=0.1, subtopic="other.general", clustering="rag.chunking") == \
        ["low_confidence", "other_general", "topic_disagrees_with_clustering"]


def test_with_every_signal_off_nothing_is_ever_flagged(tax):
    r = rule(confidence_below=0.0, other_general=False, topic_disagrees=False)
    assert reasons(r, tax, confidence=0.0, subtopic="other.general", clustering="rag.chunking") == []


# ============================================================ the command line

def test_main_reports_a_good_file(capsys):
    assert RV.main([]) == 0
    out = capsys.readouterr().out
    assert "OK: review_rule.v1.yaml (version v1)" in out and "confidence < 0.78" in out


def test_main_reports_problems_and_fails(tmp_path, capsys):
    p = write(tmp_path / "bad.yaml", {**GOOD, "confidence_below": 3})
    assert RV.main([str(p)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEMS in" in out and "confidence_below must be a number from 0 to 1" in out
