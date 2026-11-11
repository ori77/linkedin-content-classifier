"""Tests for pipeline/rubrics.py (Step 8).

Pattern: load the real rubrics file, break exactly one thing, and check the validator
complains about exactly that thing. If the validator stops noticing a break, a test fails.
"""
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import rubrics as R  # noqa: E402

RUBRICS_FILE = ROOT / "rubrics" / "rubrics.v1.yaml"
SCRIPT = ROOT / "pipeline" / "rubrics.py"


@pytest.fixture
def good():
    """A fresh copy of the real rubrics data that a test may damage."""
    return copy.deepcopy(yaml.safe_load(RUBRICS_FILE.read_text(encoding="utf-8")))


def problems_with(data, fragment):
    found = R.validate_rubrics(data)
    return [p for p in found if fragment in p]


def alignment(data, name):
    return next(a for a in data["alignment"] if a["id"] == name)


# ------------------------------------------------------------ the real file

def test_real_file_has_no_problems(good):
    assert R.validate_rubrics(good) == []


def test_load_rubrics_returns_data_and_sha():
    rub = R.load_rubrics(RUBRICS_FILE)
    assert rub.version == "v1"
    assert len(rub.sha) == 64


def test_default_path_points_at_real_file():
    assert R.DEFAULT_PATH == RUBRICS_FILE
    assert R.DEFAULT_PATH.exists()


# ------------------------------------------------------------ numbers copied from the LLD
# These lock the values that come from LLD-part-2 (9.1 to 9.4) so they cannot drift silently.

def test_lld_alignment_weights_and_bands(good):
    weights = {a["id"]: a["weight"] for a in good["alignment"]}
    assert weights == {"applicability": 0.30, "technical_depth": 0.20, "hands_on": 0.20,
                       "currency": 0.10, "career_value": 0.10, "production_relevance": 0.10}
    assert [b["min"] for b in good["alignment_bands"]] == [9.0, 7.0, 4.0, 0.0]


def test_lld_time_numbers(good):
    t = good["time"]
    assert (t["reading_wpm"], t["technical_wpm"]) == (230, 180)
    assert t["cover_clamp"] == {"low": 0.5, "high": 3.0}
    assert t["understand"]["depth_multiplier"] == {
        "beginner": 1.5, "intermediate": 2.5, "advanced": 4.0, "expert": 6.0}
    # LLD-part-2 section 9.2: TYPE = {tutorial 1.2, mini_project 1.3, full_project 1.5,
    # interview_question_set 2.0, news 1.0, miscellaneous 1.0}
    assert t["understand"]["type_multiplier"] == {
        "tutorial": 1.2, "mini_project": 1.3, "full_project": 1.5,
        "interview_question_set": 2.0, "news": 1.0, "miscellaneous": 1.0}
    assert t["buckets"]["edges_minutes"] == [15, 60, 180, 480, 1440]


# ------------------------------------------------------------ top level

def test_top_level_must_be_mapping():
    assert R.validate_rubrics(["not", "a", "mapping"])


@pytest.mark.parametrize("key", ["version", "role_profile", "judging_basis"])
def test_empty_top_level_text(good, key):
    good[key] = "  "
    assert problems_with(good, f"'{key}' must be a non-empty text")


# ------------------------------------------------------------ difficulty

def test_difficulty_missing_level(good):
    good["difficulty"] = [d for d in good["difficulty"] if d["id"] != "expert"]
    assert problems_with(good, "difficulty: missing ids ['expert']")


def test_difficulty_unknown_level(good):
    good["difficulty"][0]["id"] = "novice"
    assert problems_with(good, "unknown ids ['novice']")


def test_difficulty_wrong_order(good):
    good["difficulty"].reverse()
    assert problems_with(good, "difficulty: ids must be in the order")


def test_difficulty_duplicate(good):
    good["difficulty"][1]["id"] = "beginner"
    assert problems_with(good, "difficulty: duplicate ids")


@pytest.mark.parametrize("field", ["definition", "example"])
def test_difficulty_empty_field(good, field):
    good["difficulty"][2][field] = ""
    assert problems_with(good, f"'{field}' is empty")


def test_difficulty_item_without_id(good):
    good["difficulty"][0] = {"definition": "x"}
    assert problems_with(good, "difficulty[0]: needs an 'id'")


# ------------------------------------------------------------ content type

def test_content_type_missing(good):
    good["content_type"] = good["content_type"][:-2] + good["content_type"][-1:]
    assert problems_with(good, "content_type: missing ids")


def test_content_type_catch_all_must_be_last(good):
    good["content_type"].insert(0, good["content_type"].pop())  # move miscellaneous first
    assert problems_with(good, "must be the last rule")


def test_content_type_empty_rule(good):
    good["content_type"][0]["rule"] = ""
    assert problems_with(good, "'rule' is empty")


# ------------------------------------------------------------ alignment

def test_alignment_missing_criterion(good):
    good["alignment"] = good["alignment"][1:]
    assert problems_with(good, "alignment: missing ids")


def test_alignment_weights_must_sum_to_one(good):
    alignment(good, "applicability")["weight"] = 0.35
    assert problems_with(good, "weights must sum to 1")


def test_alignment_weights_that_sum_but_one_is_zero(good):
    alignment(good, "applicability")["weight"] = 0.0
    alignment(good, "technical_depth")["weight"] = 0.5
    assert problems_with(good, "weight must be a number above 0")


def test_alignment_weight_not_a_number(good):
    alignment(good, "currency")["weight"] = "0.1"
    assert problems_with(good, "weight must be a number above 0")


@pytest.mark.parametrize("field", ["question", "high", "low"])
def test_alignment_empty_text(good, field):
    alignment(good, "hands_on")[field] = ""
    assert problems_with(good, f"'{field}' is empty")


# ------------------------------------------------------------ bands

def test_bands_must_decrease(good):
    good["alignment_bands"][1]["min"] = 9.5
    assert problems_with(good, "must strictly decrease")


def test_bands_equal_mins_rejected(good):
    good["alignment_bands"][1]["min"] = good["alignment_bands"][0]["min"]  # two bands, same start
    assert problems_with(good, "must strictly decrease")


def test_bands_must_reach_zero(good):
    good["alignment_bands"][-1]["min"] = 1.0
    assert problems_with(good, "must start at 0")


def test_bands_min_out_of_range(good):
    good["alignment_bands"][0]["min"] = 11
    assert problems_with(good, "from 0 to 10")


def test_bands_empty_label(good):
    good["alignment_bands"][0]["label"] = ""
    assert problems_with(good, "'label' is empty")


def test_bands_empty_list(good):
    good["alignment_bands"] = []
    assert problems_with(good, "needs at least one band")


# ------------------------------------------------------------ time: speeds and basis

def test_technical_wpm_not_faster_than_reading(good):
    good["time"]["technical_wpm"] = 300
    assert problems_with(good, "must not be faster")


def test_wpm_must_be_positive(good):
    good["time"]["reading_wpm"] = 0
    assert problems_with(good, "time.reading_wpm must be a number above 0")


def test_basis_missing(good):
    good["time"]["basis"] = good["time"]["basis"][:-1]
    assert problems_with(good, "time.basis: missing ids")


def test_basis_empty_rule(good):
    good["time"]["basis"][0]["rule"] = ""
    assert problems_with(good, "time.basis")


# ------------------------------------------------------------ time: hands-on and hints

def test_hands_on_floor_missing_type(good):
    del good["time"]["hands_on_floor_minutes"]["tutorial"]
    assert problems_with(good, "hands_on_floor_minutes: missing ids")


def test_hands_on_floor_negative(good):
    good["time"]["hands_on_floor_minutes"]["mini_project"] = -5
    assert problems_with(good, "must be a number, 0 or more")


def test_hands_on_applies_when_unknown_basis(good):
    good["time"]["hands_on_applies_when_basis"] = ["made_up"]
    assert problems_with(good, "hands_on_applies_when_basis")


def test_hands_on_applies_when_empty_list(good):
    good["time"]["hands_on_applies_when_basis"] = []
    assert problems_with(good, "hands_on_applies_when_basis")


def test_true_is_not_a_number(good):
    # in Python True == 1, but a YAML 'true' where a number belongs is a mistake
    good["time"]["understand"]["news_cap_vs_cover"] = True
    assert problems_with(good, "news_cap_vs_cover")


# ------------------------------------------------------------ time: stated units (Step 9)

def unit(data, name):
    return next(u for u in data["time"]["stated_units"] if u["id"] == name)


def test_stated_units_values(good):
    per = {u["id"]: u.get("minutes_per_unit") for u in good["time"]["stated_units"]}
    assert per == {"minutes": 1, "hours": 60, "weeks": 300, "questions": 3, "other": None, "none": None}


def test_stated_units_must_be_a_list(good):
    good["time"]["stated_units"] = {"minutes": 1}
    assert problems_with(good, "time.stated_units must be a list")


def test_stated_units_missing_key(good):
    del good["time"]["stated_units"]
    assert problems_with(good, "time.stated_units must be a list")


def test_stated_units_missing_unit(good):
    good["time"]["stated_units"] = [u for u in good["time"]["stated_units"] if u["id"] != "hours"]
    assert problems_with(good, "stated_units: missing ids")


def test_stated_units_unknown_unit(good):
    good["time"]["stated_units"].append({"id": "fortnights", "minutes_per_unit": 1, "describe": "x"})
    assert problems_with(good, "stated_units: unknown ids")


def test_stated_units_duplicate_unit(good):
    good["time"]["stated_units"].append(copy.deepcopy(unit(good, "hours")))
    assert problems_with(good, "stated_units: duplicate ids")


@pytest.mark.parametrize("bad", [0, -5, "60", None, True])
def test_stated_unit_minutes_must_be_a_positive_number(good, bad):
    unit(good, "hours")["minutes_per_unit"] = bad
    assert problems_with(good, "'hours': minutes_per_unit must be a number above 0")


@pytest.mark.parametrize("name", ["other", "none"])
def test_unconvertible_units_must_not_have_minutes(good, name):
    unit(good, name)["minutes_per_unit"] = 10
    assert problems_with(good, f"'{name}': must not have minutes_per_unit")


def test_stated_unit_needs_a_description(good):
    unit(good, "weeks")["describe"] = "  "
    assert problems_with(good, "'weeks': 'describe' is empty")


def test_stated_unit_without_id(good):
    good["time"]["stated_units"].append({"minutes_per_unit": 2, "describe": "x"})
    assert problems_with(good, "stated_units[6]: needs an 'id'")


# ------------------------------------------------------------ time: clamps

def test_cover_clamp_low_above_one(good):
    good["time"]["cover_clamp"]["low"] = 1.5
    assert problems_with(good, "time.cover_clamp must satisfy")


def test_cover_clamp_high_below_one(good):
    good["time"]["cover_clamp"]["high"] = 0.9
    assert problems_with(good, "time.cover_clamp must satisfy")


def test_cover_clamp_applies_when_unknown_basis(good):
    good["time"]["cover_clamp_applies_when_basis"] = ["made_up"]
    assert problems_with(good, "cover_clamp_applies_when_basis")


def test_cover_clamp_applies_when_empty(good):
    good["time"]["cover_clamp_applies_when_basis"] = []
    assert problems_with(good, "cover_clamp_applies_when_basis")


def test_cover_clamp_applies_when_missing(good):
    del good["time"]["cover_clamp_applies_when_basis"]
    assert problems_with(good, "cover_clamp_applies_when_basis")


def test_limits_cover_above_understand(good):
    good["time"]["limits_minutes"]["cover_max"] = 99999999
    assert problems_with(good, "cover_max <= understand_max")


# ------------------------------------------------------------ time: understand

def test_depth_multiplier_missing_level(good):
    del good["time"]["understand"]["depth_multiplier"]["expert"]
    assert problems_with(good, "depth_multiplier: missing ids")


def test_depth_multiplier_must_not_decrease(good):
    good["time"]["understand"]["depth_multiplier"]["advanced"] = 2.0  # below intermediate 2.5
    assert problems_with(good, "must not decrease with difficulty")


def test_depth_multiplier_must_be_positive(good):
    good["time"]["understand"]["depth_multiplier"]["beginner"] = 0
    assert problems_with(good, "every value must be a number above 0")


def test_type_multiplier_missing_type(good):
    del good["time"]["understand"]["type_multiplier"]["news"]
    assert problems_with(good, "type_multiplier: missing ids")


def test_type_multiplier_must_be_positive(good):
    good["time"]["understand"]["type_multiplier"]["news"] = -1
    assert problems_with(good, "type_multiplier: every value")


def test_combine_only_geometric_mean(good):
    good["time"]["understand"]["combine"] = "arithmetic_mean"
    assert problems_with(good, "geometric_mean")


def test_clamp_vs_cover_low_below_one(good):
    good["time"]["understand"]["clamp_vs_cover"]["low"] = 0.8
    assert problems_with(good, "understand >= cover")


def test_clamp_vs_cover_high_not_above_low(good):
    good["time"]["understand"]["clamp_vs_cover"]["high"] = 1.0
    assert problems_with(good, "clamp_vs_cover.high must be above low")


def test_news_cap_below_one(good):
    good["time"]["understand"]["news_cap_vs_cover"] = 0.5
    assert problems_with(good, "news_cap_vs_cover")


# ------------------------------------------------------------ time: rounding and buckets

def test_round_nice_below_must_increase(good):
    good["time"]["round_nice"][1]["below"] = 30  # first entry has below 60
    assert problems_with(good, "must strictly increase")


def test_round_nice_last_has_no_below(good):
    good["time"]["round_nice"][-1]["below"] = 9999
    assert problems_with(good, "last entry must have no 'below'")


def test_round_nice_step_positive(good):
    good["time"]["round_nice"][0]["step"] = 0
    assert problems_with(good, "'step' must be a number above 0")


def test_bucket_edges_must_increase(good):
    good["time"]["buckets"]["edges_minutes"] = [15, 60, 60, 480, 1440]
    assert problems_with(good, "edges_minutes must be a strictly increasing")


def test_bucket_label_count(good):
    good["time"]["buckets"]["labels"] = good["time"]["buckets"]["labels"][:-1]
    assert problems_with(good, "one more entry than edges_minutes")


def test_bucket_labels_unique(good):
    good["time"]["buckets"]["labels"][1] = good["time"]["buckets"]["labels"][0]
    assert problems_with(good, "non-empty and unique")


# ------------------------------------------------------------ summary

def test_summary_target_outside_hard_limits(good):
    good["summary"]["target_words"]["max"] = good["summary"]["hard_words"]["max"] + 1
    assert problems_with(good, "summary: need 0 < hard.min")


def test_summary_single_paragraph_must_be_bool(good):
    good["summary"]["single_paragraph"] = "yes"
    assert problems_with(good, "single_paragraph")


@pytest.mark.parametrize("key", ["must_cover", "motives"])
def test_summary_lists_not_empty(good, key):
    good["summary"][key] = []
    assert problems_with(good, f"summary.{key}")


def test_missing_section_reported(good):
    del good["summary"]
    assert problems_with(good, "missing or wrong type: 'summary'")


# ------------------------------------------------------------ all problems together

def test_all_problems_reported_together(good):
    good["difficulty"].reverse()
    alignment(good, "applicability")["weight"] = 0.9
    good["time"]["technical_wpm"] = 999
    found = R.validate_rubrics(good)
    assert len(found) >= 3


# ------------------------------------------------------------ fingerprint

def test_sha_ignores_windows_line_endings():
    unix = b"a: 1\nb: 2\n"
    windows = b"a: 1\r\nb: 2\r\n"
    assert R.rubrics_sha(unix) == R.rubrics_sha(windows)


def test_sha_changes_when_a_value_changes():
    assert R.rubrics_sha(b"weight: 0.30\n") != R.rubrics_sha(b"weight: 0.31\n")


def test_sha_matches_the_real_file_bytes():
    raw = RUBRICS_FILE.read_bytes()
    assert R.load_rubrics(RUBRICS_FILE).sha == R.rubrics_sha(raw)


def test_crlf_copy_of_real_file_loads_with_same_sha(tmp_path):
    crlf = tmp_path / "r.yaml"
    crlf.write_bytes(RUBRICS_FILE.read_bytes().replace(b"\n", b"\r\n"))
    assert R.load_rubrics(crlf).sha == R.load_rubrics(RUBRICS_FILE).sha


# ------------------------------------------------------------ loading errors

def test_load_missing_file(tmp_path):
    with pytest.raises(R.RubricsError, match="cannot read"):
        R.load_rubrics(tmp_path / "nope.yaml")


def test_load_bad_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(R.RubricsError, match="not valid YAML"):
        R.load_rubrics(bad)


def test_load_invalid_rubrics_lists_problems(tmp_path, good):
    good["alignment"][0]["weight"] = 0.9
    f = tmp_path / "r.yaml"
    f.write_text(yaml.safe_dump(good), encoding="utf-8")
    with pytest.raises(R.RubricsError, match="weights must sum to 1"):
        R.load_rubrics(f)


# ------------------------------------------------------------ command line

def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_cli_ok_on_real_file():
    r = run_cli()
    assert r.returncode == 0
    assert r.stdout.startswith("OK:")


def test_cli_exit_1_and_lists_all_problems(tmp_path, good):
    good["difficulty"].reverse()
    alignment(good, "applicability")["weight"] = 0.9
    f = tmp_path / "r.yaml"
    f.write_text(yaml.safe_dump(good), encoding="utf-8")
    r = run_cli(str(f))
    assert r.returncode == 1
    assert "order" in r.stdout and "sum to 1" in r.stdout


def test_cli_exit_1_on_missing_file(tmp_path):
    r = run_cli(str(tmp_path / "nope.yaml"))
    assert r.returncode == 1
    assert "cannot read" in r.stdout
