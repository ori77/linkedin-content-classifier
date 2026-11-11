"""Tests for pipeline/scoring.py (Step 9, part 1: the arithmetic).

Expected values were worked out by hand from the rules in rubrics.v1.yaml (see comments),
not copied from the code's output.
"""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import rubrics as R  # noqa: E402
import scoring as S  # noqa: E402

RUB = R.load_rubrics(ROOT / "rubrics" / "rubrics.v1.yaml")


def fin(**kw):
    base = dict(baseline=4, llm_cover=6, llm_understand=20, difficulty="intermediate",
                content_type="tutorial", basis="post_only", rub=RUB)
    base.update(kw)
    return S.finalize_time(**base)


def rub_with_alignment(**weights):
    data = copy.deepcopy(RUB.data)
    for item in data["alignment"]:
        item["weight"] = weights.get(item["id"], item["weight"])
    return R.Rubrics(data=data, sha="x", path=Path("x"))


# ------------------------------------------------------------ baseline reading time

@pytest.mark.parametrize("words,expected", [(0, 1), (1, 1), (230, 1), (231, 2), (460, 2), (461, 3)])
def test_baseline_prose(words, expected):
    assert S.baseline_cover_minutes(words, RUB) == expected      # 230 words per minute, rounded up


@pytest.mark.parametrize("words,expected", [(180, 1), (181, 2), (360, 2), (361, 3)])
def test_baseline_technical_is_slower(words, expected):
    assert S.baseline_cover_minutes(words, RUB, technical=True) == expected   # 180 per minute


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), None, "10", True])
def test_baseline_rejects_bad_word_counts(bad):
    with pytest.raises(ValueError):
        S.baseline_cover_minutes(bad, RUB)


# ------------------------------------------------------------ round_nice and buckets

@pytest.mark.parametrize("minutes,expected", [
    (0.4, 5),     # never below one step
    (2, 5), (7, 5), (7.5, 10), (8, 10),      # nearest 5 below 60 (half goes up)
    (57.4, 55), (57.5, 60),
    (62, 60), (67, 60), (68, 75), (67.5, 75),  # nearest 15 from 60 to 480
    (479, 480), (480, 480),
    (500, 480), (509, 480), (510, 540),      # nearest 60 from 480
    (8820.0, 8820),
])
def test_round_nice(minutes, expected):
    assert S.round_nice(minutes, RUB) == expected


@pytest.mark.parametrize("bad", [0, -5, float("nan"), None, True])
def test_round_nice_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        S.round_nice(bad, RUB)


@pytest.mark.parametrize("minutes,label", [
    (1, "<15m"), (14.9, "<15m"), (15, "15-60m"), (59.9, "15-60m"), (60, "1-3h"),
    (179, "1-3h"), (180, "3-8h"), (479, "3-8h"), (480, "1-3d"), (1439, "1-3d"),
    (1440, ">3d"), (20160, ">3d"),
])
def test_bucket_edges_belong_to_the_bucket_above(minutes, label):
    assert S.bucket(minutes, RUB) == label


# ------------------------------------------------------------ finalize_time: cover

def test_post_only_worked_example():
    # baseline 4, model says 6 (inside 2..12) -> 6 -> shown as 5.
    # understand: 6 x 2.5 (intermediate) x 1.2 (tutorial) = 18; geometric mean with 20 = 18.97 -> 20
    r = fin()
    assert (r["cover_minutes"], r["understand_minutes"]) == (5, 20)
    assert (r["cover_bucket"], r["understand_bucket"]) == ("<15m", "15-60m")
    assert r["baseline_cover_minutes"] == 4 and r["basis"] == "post_only"


def test_post_only_model_cannot_go_above_3x_baseline():
    assert fin(llm_cover=100)["cover_minutes"] == 10        # clamp to 12 -> shown as 10


def test_post_only_model_cannot_go_below_half_baseline():
    assert fin(baseline=40, llm_cover=1)["cover_minutes"] == 20   # clamp to 20


def test_post_only_has_no_hands_on_floor():
    # a mini-project judged from the post alone: floor 45 must NOT be added (ref stays 4)
    assert fin(content_type="mini_project", llm_cover=8)["cover_minutes"] == 10


def test_stated_resource_is_not_crushed_by_the_post_reading_time():
    # the motivating case: a 60-hour course advertised in a 2-minute post
    r = fin(baseline=2, llm_cover=3600, llm_understand=7200, basis="stated_in_post", stated=3600)
    assert r["cover_minutes"] == 3600 and r["cover_bucket"] == ">3d"
    # understand: 3600 x 2.5 x 1.2 = 10800; sqrt(7200 x 10800) = 8818.2 -> 8820
    assert r["understand_minutes"] == 8820


def test_stated_full_project_worked_example():
    # 3600 x 4.0 (advanced) x 1.5 (full project) = 21600; sqrt(7200 x 21600) = 12470.9 -> 12480
    r = fin(baseline=2, llm_cover=3600, llm_understand=7200, difficulty="advanced",
            content_type="full_project", basis="stated_in_post", stated=3600)
    assert (r["cover_minutes"], r["understand_minutes"]) == (3600, 12480)


@pytest.mark.parametrize("basis", ["stated_in_post", "estimated_from_post"])
def test_hands_on_floor_applies_to_stated_and_estimated(basis):
    r = fin(baseline=2, llm_cover=10, llm_understand=30, content_type="mini_project", basis=basis,
            stated=10 if basis == "stated_in_post" else None)
    assert r["cover_minutes"] == 45                          # raised to the mini-project floor
    # understand: 45 x 2.5 x 1.3 = 146.25; sqrt(30 x 146.25) = 66.2 -> nearest 15 = 60
    assert r["understand_minutes"] == 60


def test_full_project_floor_is_180():
    r = fin(baseline=2, llm_cover=20, content_type="full_project", basis="estimated_from_post")
    assert r["cover_minutes"] == 180


def test_cover_has_an_upper_limit_of_14_days():
    r = fin(llm_cover=10**7, llm_understand=10**9, difficulty="expert", content_type="full_project",
            basis="stated_in_post", stated=10**7)
    assert r["cover_minutes"] == 20160                       # 14 days
    assert r["understand_minutes"] == 43200                  # 30 days


# ------------------------------------------------------------ stated size -> minutes (Step 9)

@pytest.mark.parametrize("count,unit,expected", [
    (20, "questions", 60),          # 20 x 3
    (5, "hours", 300),              # 5 x 60
    (8, "weeks", 2400),             # 8 x 300 (5 hours a week)
    (90, "minutes", 90),
    (1.5, "hours", 90),
])
def test_stated_minutes_multiplies_count_by_the_unit(count, unit, expected):
    assert S.stated_minutes(count, unit, RUB) == expected


@pytest.mark.parametrize("count,unit", [
    (22, "other"), (22, "none"),                       # no known duration
    (0, "hours"), (-3, "hours"), (True, "hours"),      # not a usable number
    (float("nan"), "hours"), ("5", "hours"), (None, "hours"),
])
def test_stated_minutes_is_none_when_it_cannot_be_worked_out(count, unit):
    assert S.stated_minutes(count, unit, RUB) is None


def test_stated_minutes_unknown_unit_raises():
    with pytest.raises(ValueError, match="unknown stated unit"):
        S.stated_minutes(5, "fortnights", RUB)


def test_stated_minutes_follows_the_rubrics_numbers():
    data = copy.deepcopy(RUB.data)
    next(u for u in data["time"]["stated_units"] if u["id"] == "questions")["minutes_per_unit"] = 7
    assert S.stated_minutes(10, "questions", R.Rubrics(data=data, sha=RUB.sha, path=RUB.path)) == 70


def test_stated_size_replaces_the_models_cover_guess():
    # the motivating case: 20 interview questions = 60 minutes, whatever the model guessed
    low = fin(baseline=2, llm_cover=15, llm_understand=45, content_type="interview_question_set",
              basis="stated_in_post", stated=60)
    high = fin(baseline=2, llm_cover=900, llm_understand=45, content_type="interview_question_set",
               basis="stated_in_post", stated=60)
    assert low["cover_minutes"] == high["cover_minutes"] == 60 and low["cover_bucket"] == "1-3h"
    # understand: 60 x 2.5 x 2.0 = 300; sqrt(45 x 300) = 116.2 -> nearest 15 = 120
    assert low["understand_minutes"] == high["understand_minutes"] == 120
    assert low["stated_minutes"] == 60


def test_stated_basis_needs_the_stated_minutes():
    with pytest.raises(ValueError, match="needs `stated`"):
        fin(basis="stated_in_post")


@pytest.mark.parametrize("bad", [0, -5, True, float("nan")])
def test_stated_minutes_must_be_a_positive_number(bad):
    with pytest.raises(ValueError, match="needs `stated`"):
        fin(basis="stated_in_post", stated=bad)


@pytest.mark.parametrize("basis", ["post_only", "estimated_from_post"])
def test_stated_minutes_only_belong_to_the_stated_basis(basis):
    with pytest.raises(ValueError, match="only for basis stated_in_post"):
        fin(basis=basis, stated=60)


def test_other_bases_report_no_stated_minutes():
    assert fin(basis="post_only")["stated_minutes"] is None
    assert fin(basis="estimated_from_post")["stated_minutes"] is None


# ------------------------------------------------------------ finalize_time: understand

def test_news_understand_is_capped_at_1_5x_cover():
    # baseline 3, model 10 -> clamped to 9. heuristic 9 x 2.5 x 1.0 = 22.5; sqrt(1000 x 22.5) = 150,
    # clamp to 10x cover = 90, news cap 1.5 x 9 = 13.5 -> shown as 15
    r = fin(baseline=3, llm_cover=10, llm_understand=1000, content_type="news")
    assert (r["cover_minutes"], r["understand_minutes"]) == (10, 15)


def test_understand_is_never_below_cover():
    # tiny model guess of 1 minute: sqrt(1 x 48) = 6.9, raised to the cover time (10)
    r = fin(baseline=10, llm_cover=10, llm_understand=1, difficulty="advanced")
    assert r["understand_minutes"] >= r["cover_minutes"]
    assert r["understand_minutes"] == 10


def test_understand_is_at_most_10x_cover():
    r = fin(baseline=60, llm_cover=60, llm_understand=10**6, difficulty="expert", content_type="tutorial")
    assert r["understand_minutes"] == 600                    # 10 x 60, already a tidy number


def test_understand_cap_is_applied_before_rounding():
    # the raw cap is 10 x 10 = 100 minutes; 100 is then shown as the nearest 15 = 105
    r = fin(baseline=10, llm_cover=10, llm_understand=10**6, difficulty="expert", content_type="tutorial")
    assert r["understand_minutes"] == 105


def test_harder_difficulty_means_longer_understanding():
    easy = fin(difficulty="beginner", llm_understand=30)["understand_minutes"]
    hard = fin(difficulty="expert", llm_understand=30)["understand_minutes"]
    assert hard > easy


def test_type_multiplier_is_used():
    # interview sets (2.0) take longer to understand than news (1.0) for the same inputs
    a = fin(baseline=30, llm_cover=30, llm_understand=60, content_type="interview_question_set")
    b = fin(baseline=30, llm_cover=30, llm_understand=60, content_type="miscellaneous")
    assert a["understand_minutes"] > b["understand_minutes"]


def test_bucket_matches_the_displayed_number():
    # raw cover 14.0 is shown as 15; its bucket must be '15-60m', not '<15m'
    r = fin(baseline=28, llm_cover=14)
    assert r["cover_minutes"] == 15 and r["cover_bucket"] == "15-60m"


@pytest.mark.parametrize("field,value", [("difficulty", "easy"), ("content_type", "blog"), ("basis", "guess")])
def test_unknown_labels_rejected(field, value):
    with pytest.raises(ValueError):
        fin(**{field: value})


@pytest.mark.parametrize("field", ["baseline", "llm_cover", "llm_understand"])
@pytest.mark.parametrize("bad", [0, -3, float("nan"), None])
def test_bad_numbers_rejected(field, bad):
    with pytest.raises(ValueError):
        fin(**{field: bad})


# ------------------------------------------------------------ alignment

def scores(**kw):
    base = dict(applicability=5, technical_depth=5, hands_on=5, currency=5, career_value=5,
                production_relevance=5)
    base.update(kw)
    return base


def test_alignment_worked_example():
    # 0.3x8 + 0.2x6 + 0.2x9 + 0.1x7 + 0.1x5 + 0.1x4 = 2.4 + 1.2 + 1.8 + 0.7 + 0.5 + 0.4 = 7.0
    s = scores(applicability=8, technical_depth=6, hands_on=9, currency=7, career_value=5,
               production_relevance=4)
    assert S.alignment_score(s, RUB) == 7.0


def test_alignment_extremes():
    assert S.alignment_score({k: 10 for k in scores()}, RUB) == 10.0
    assert S.alignment_score({k: 0 for k in scores()}, RUB) == 0.0


def test_alignment_uses_each_weight():
    # raising applicability by 1 adds 0.3; raising production_relevance by 1 adds 0.1
    base = S.alignment_score(scores(), RUB)
    assert S.alignment_score(scores(applicability=6), RUB) == pytest.approx(base + 0.3)
    assert S.alignment_score(scores(production_relevance=6), RUB) == pytest.approx(base + 0.1)
    assert S.alignment_score(scores(hands_on=6), RUB) == pytest.approx(base + 0.2)


def test_alignment_halves_round_up():
    # custom weights so a score of exactly 0.15 and 0.05 can occur (the real weights give only
    # multiples of 0.1). Plain round(0.15, 1) gives 0.1 because of float error; we want 0.2.
    rub = rub_with_alignment(currency=0.15, career_value=0.05, production_relevance=0.05, hands_on=0.15)
    only = lambda **kw: {k: kw.get(k, 0) for k in scores()}   # noqa: E731
    assert S.alignment_score(only(currency=1), rub) == 0.2
    assert S.alignment_score(only(career_value=1), rub) == 0.1


@pytest.mark.parametrize("bad", [
    {"applicability": 5},                                      # missing keys
    {**scores(), "extra": 1},                                  # unknown key
    scores(applicability=11), scores(applicability=-1),        # out of range
    scores(applicability=float("nan")), scores(applicability="8"), scores(applicability=True),
])
def test_alignment_rejects_bad_subscores(bad):
    with pytest.raises(ValueError):
        S.alignment_score(bad, RUB)


@pytest.mark.parametrize("score,label", [
    (10.0, "Must-read / must-build for an AI Engineer"), (9.0, "Must-read / must-build for an AI Engineer"),
    (8.9, "Highly relevant"), (7.0, "Highly relevant"),
    (6.9, "Useful background"), (4.0, "Useful background"),
    (3.9, "Low relevance (off-topic, fluff, or unrelated)"), (0.0, "Low relevance (off-topic, fluff, or unrelated)"),
])
def test_alignment_bands(score, label):
    assert S.alignment_band(score, RUB) == label


@pytest.mark.parametrize("bad", [-0.1, 10.1, float("nan"), None])
def test_alignment_band_rejects_bad_scores(bad):
    with pytest.raises(ValueError):
        S.alignment_band(bad, RUB)
