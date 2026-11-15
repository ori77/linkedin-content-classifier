"""A small made-up database for the dashboard tests (not a test file: pytest does not collect it).

Built with the real loader (step15_load_sqlite.py), so the tables and the library view are the
real ones. Seven posts in the CSV, six classified:

  id  title                  subtopic           difficulty    type                    score  conf  flagged  links
  1   Chunking guide         rag.chunking       beginner      tutorial                10.0   0.98  no       3
  2   Evaluation project     rag.evaluation     advanced      full_project            6.0    0.96  no       0
  3   Interview prep         careers.interviews intermediate  interview_question_set  3.0    0.61  yes      1   (clustering says another topic)
  4   Cheap tricks           rag.chunking       beginner      miscellaneous           5.0    0.75  yes      0   (clustering says another topic)
  5   Odd one                other.general      beginner      miscellaneous           1.0    0.93  yes      0   (other.general; no save date)
  6   Tips & $tricks* 100%   rag.evaluation     intermediate  tutorial                7.0    0.67  yes      2   (same topic, other subtopic)
  7   (no text)              not classified, but it keeps its post link

Review queue order (lowest confidence first): 3, 6, 4, 5.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

import step15_load_sqlite as L

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
RULE = {"version": "v1", "confidence_below": 0.78, "other_general": True, "topic_disagrees_with_clustering": True}
COLUMNS = ["activity_id", "saved_at", "post_url", "post_date_utc", "author", "text", "text_clean", "urls",
           "attachment_title", "attachment_site", "has_image", "source_file", "status"]


def url_of(aid: int) -> str:
    return f"https://www.linkedin.com/feed/update/urn:li:activity:{aid}"


# (id, title/first line, author, saved_at, urls)
POSTS = [
    (1, "Chunking guide", "Ada", "2026-10-01 10:00:00", "https://lnkd.in/a1 | https://example.com/two | https://lnkd.in/a3"),
    (2, "Evaluation project", "Bob", "2026-10-05 10:00:00", ""),
    (3, "Interview prep", "Cy", "2026-09-01 10:00:00", "https://lnkd.in/c1"),
    (4, "Cheap tricks", "Di", "2026-08-01 10:00:00", ""),
    (5, "Odd one", "Ed", "", ""),
    (6, "Tips & $tricks* 100%", "Fay", "2026-07-01 10:00:00", "https://lnkd.in/f1 | https://example.com/f(2)"),
    (7, "", "Gus", "2026-06-01 10:00:00", ""),
]
# id: (subtopic, clustering, difficulty, type, score, band, cover, understand, cover_bucket, understand_bucket, self_conf)
LABELS = {
    1: ("rag.chunking", "rag.chunking", "beginner", "tutorial", 10.0, "Must-read", 30, 90, "15-60m", "1-3h", 0.95),
    2: ("rag.evaluation", "rag.evaluation", "advanced", "full_project", 6.0, "Highly relevant", 600, 1500, "3-8h", "1-3d", 0.9),
    3: ("careers.interviews", "rag.chunking", "intermediate", "interview_question_set", 3.0, "Low relevance", 40, 40, "15-60m", "15-60m", 0.6),
    4: ("rag.chunking", "careers.interviews", "beginner", "miscellaneous", 5.0, "Useful background", 5, 5, "<15m", "<15m", 0.9),
    5: ("other.general", "other.general", "beginner", "miscellaneous", 1.0, "Low relevance", 5, 5, "<15m", "<15m", 0.9),
    6: ("rag.evaluation", "rag.chunking", "intermediate", "tutorial", 7.0, "Highly relevant", 200, 300, "3-8h", "3-8h", 0.5),
}


def make_record(aid: int, **over) -> dict:
    sub, clustering, diff, ctype, score, band, cover, understand, cb, ub, self_conf = LABELS[aid]
    rec = {
        "activity_id": aid, "topic_id": "other" if sub == "other.general" else sub.split(".")[0], "subtopic_id": sub,
        "difficulty": diff, "content_type": ctype,
        "time": {"cover_minutes": cover, "understand_minutes": understand, "cover_bucket": cb,
                 "understand_bucket": ub, "basis": "post_only", "stated_minutes": None, "stated_size": ""},
        "alignment_score": score, "alignment_band": band, "alignment_subscores": {"applicability": 7},
        "summary": f"Summary of post {aid}.", "key_points": ["first point", "second point"],
        "rationales": {"topic": f"topic reason {aid}", "content_type": f"type reason {aid}",
                       "difficulty": f"difficulty reason {aid}", "time": f"time reason {aid}",
                       "alignment": f"alignment reason {aid}"},
        "confidence": 0.5, "confidence_parts": {"self": self_conf, "agree": 1.0, "valid": 1.0, "partial": False, "penalties": 0},
        "needs_review": False, "review_reasons": [], "review_rule_version": "v1", "review_rule_sha": "x" * 64,
        "warnings": ["interview_mentioned"] if aid == 3 else [], "clustering_subtopic_id": clustering,
        "model": "gemma4:e4b", "prompt_version": "p1", "prompt_sha": "p" * 64, "taxonomy_version": "v1",
        "rubrics_version": "v2", "rubrics_sha": "r" * 64, "cache_key": f"key-{aid}-" + "0" * 40,
        "attempts": {"topic": 1, "rest": 1}, "retry_reasons": {"topic": [], "rest": []},
        "input_tokens": 100, "output_tokens": 50, "seconds": 10.0, "created_at": "2026-11-13T03:00:00+00:00",
    }
    rec.update(over)
    return rec


def build(directory: Path) -> Path:
    """Create the database in `directory` and return its path."""
    tax = directory / "tax.yaml"
    tax.write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    rule = directory / "rule.yaml"
    rule.write_text(yaml.safe_dump(RULE), encoding="utf-8")
    csv_path = directory / "posts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for aid, title, author, saved, urls in POSTS:
            text = f"{title}\nbody text of post {aid}" if title else ""
            w.writerow({"activity_id": aid, "saved_at": saved, "post_url": url_of(aid), "post_date_utc": "2026-10-01",
                        "author": author, "text": text, "text_clean": text, "urls": urls, "attachment_title": "",
                        "attachment_site": "", "has_image": "False", "source_file": "sr1.html", "status": "captured"})
    jsonl = directory / "run1.jsonl"
    jsonl.write_text("".join(json.dumps(make_record(aid)) + "\n" for aid in LABELS), encoding="utf-8")
    db = directory / "lib.sqlite"
    report = L.load(db, jsonl, csv_path, tax, rule)
    assert report.problems == [], report.problems
    return db
