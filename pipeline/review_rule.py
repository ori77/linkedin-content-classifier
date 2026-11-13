"""The review rule (Step 13): which posts go to the review queue, and why.

In simple terms: after the model has classified a post, this file decides whether a human
should look at it. A post is flagged for review when ANY of these is true. Each one can be
switched off or tuned in rubrics/review_rule.v1.yaml, with no code change:

  low_confidence                   the confidence is below `confidence_below`
  other_general                    the model put the post in "other.general" (fits no subtopic)
  topic_disagrees_with_clustering  the TOPIC the model chose differs from the topic the earlier
                                   clustering gave the post. A post the clustering could not
                                   place (it is in "other.general") counts as "no opinion".

Why this is a separate file and not part of rubrics.v2.yaml: the rule does not change what the
model is asked or what it answers, so changing it must not make old results look out of date
(the rubrics fingerprint is part of the cache key). The rule only reads fields that every
record stores (confidence, subtopic_id, clustering_subtopic_id), so it can be applied again to a
finished run at no model cost. The Step 15 loader will do that, so changing the number after
the full run never needs a re-run.

Check a file from the command line:
    python review_rule.py                      (checks rubrics/review_rule.v1.yaml)
    python review_rule.py path/to/other.yaml
Exit status 0 = fine, 1 = problems (all problems are printed together).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from classify_inputs import OTHER_ID, Taxonomy
from rubrics import rubrics_sha

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "rubrics" / "review_rule.v1.yaml"

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_OTHER = "other_general"
REASON_DISAGREES = "topic_disagrees_with_clustering"
REASONS = (REASON_LOW_CONFIDENCE, REASON_OTHER, REASON_DISAGREES)

KEYS = ("version", "confidence_below", "other_general", "topic_disagrees_with_clustering")


class ReviewRuleError(Exception):
    """Raised by load_review_rule when the file has problems; the message lists all of them."""


@dataclass(frozen=True)
class ReviewRule:
    version: str
    confidence_below: float             # flag when confidence < this (0 switches the signal off)
    other_general: bool                 # flag posts the model put in other.general
    topic_disagrees: bool               # flag posts whose topic differs from the clustering's topic
    sha: str                            # fingerprint of the file, stored in every record for audit


def validate_review_rule(data: Any) -> list[str]:
    """Every problem with the file content, as readable sentences (empty list = fine)."""
    if not isinstance(data, dict):
        return ["the file must contain a mapping at the top level"]
    problems = []
    missing = [k for k in KEYS if k not in data]
    unknown = [k for k in data if k not in KEYS]
    if missing:
        problems.append(f"missing keys: {missing}")
    if unknown:                          # a typo such as 'confidence_belov' must not be silently ignored
        problems.append(f"unknown keys: {unknown} (allowed: {list(KEYS)})")
    version = data.get("version")
    if "version" in data and (not isinstance(version, str) or not version.strip()):
        problems.append("version must be a non-empty text")
    cb = data.get("confidence_below")
    if "confidence_below" in data and (       # nan and infinity fail the range test too
            not isinstance(cb, (int, float)) or isinstance(cb, bool) or not 0 <= cb <= 1):
        problems.append(f"confidence_below must be a number from 0 to 1, got {cb!r}")
    for key in ("other_general", "topic_disagrees_with_clustering"):
        if key in data and not isinstance(data[key], bool):
            problems.append(f"{key} must be true or false, got {data[key]!r}")
    return problems


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
    return data, rubrics_sha(raw), validate_review_rule(data)


def load_review_rule(path: Path | str = DEFAULT_PATH) -> ReviewRule:
    """Load the rule or raise ReviewRuleError listing every problem."""
    data, sha, problems = check_file(path)
    if problems or data is None or sha is None:     # the None checks only help the type checker
        raise ReviewRuleError("review rule problems:\n- " + "\n- ".join(problems))
    return ReviewRule(version=data["version"], confidence_below=float(data["confidence_below"]),
                      other_general=data["other_general"],
                      topic_disagrees=data["topic_disagrees_with_clustering"], sha=sha)


def review_reasons(rule: ReviewRule, *, confidence: float, subtopic_id: str,
                   clustering_subtopic_id: str | None, taxonomy: Taxonomy) -> list[str]:
    """Why a post needs review. An empty list means it does not.

    The reasons come back in a fixed order (the order of REASONS), so a record is stable.
    """
    reasons = []
    if confidence < rule.confidence_below:
        reasons.append(REASON_LOW_CONFIDENCE)
    if rule.other_general and subtopic_id == OTHER_ID:
        reasons.append(REASON_OTHER)
    if (rule.topic_disagrees and clustering_subtopic_id not in (None, OTHER_ID)
            and taxonomy.topic_of(subtopic_id) != taxonomy.topic_of(clustering_subtopic_id)):
        reasons.append(REASON_DISAGREES)
    return reasons


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_PATH
    data, sha, problems = check_file(path)
    if problems or data is None or sha is None:     # the None checks only help the type checker
        print(f"PROBLEMS in {path}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {path.name} (version {data['version']}), sha256 {sha[:12]}")
    print(f"  flag when confidence < {data['confidence_below']}; other.general flagged: {data['other_general']}; "
          f"topic disagreement flagged: {data['topic_disagrees_with_clustering']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
