"""Read and check the three inputs of the classifier (Step 9).

In simple terms: before any model is called we make sure the files we depend on are sane,
and we turn them into small, plain Python objects.

  * load_taxonomy    : taxonomy.v1.yaml -> the list of subtopics the model may choose from
  * load_assignments : taxonomy_assignments.v1.csv -> which subtopic clustering gave each post
                       (used later as an independent opinion for the confidence score)
  * load_posts       : posts_with_save_dates.csv -> the posts to classify

Every loader collects ALL problems and reports them together (like step 5b).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

OTHER_ID = "other.general"        # how step 5b writes "Other"
OTHER_TOPIC_ID = "other"
MAX_POST_CHARS = 12000            # LinkedIn posts are far shorter; this only guards against surprises


class InputError(Exception):
    """An input file has problems; the message lists all of them."""


def file_sha(raw: bytes) -> str:
    """Fingerprint of a file's text. Windows (CRLF) and Linux (LF) copies give the same hash."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _raise(what: str, problems: list[str]) -> None:
    if problems:
        raise InputError(f"{what} problems:\n- " + "\n- ".join(problems))


# ------------------------------------------------------------------ taxonomy

@dataclass(frozen=True)
class Subtopic:
    id: str
    name: str
    topic_id: str
    topic_name: str
    description: str
    keywords: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.topic_name} > {self.name}"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    subtopics: tuple[Subtopic, ...]       # the real subtopics, in file order (no "other")
    sha: str
    path: Path
    by_id: dict[str, Subtopic] = field(default_factory=dict, compare=False)

    @property
    def choice_ids(self) -> list[str]:
        """Everything the model may answer: every subtopic id, then other.general."""
        return [s.id for s in self.subtopics] + [OTHER_ID]

    def topic_of(self, subtopic_id: str) -> str:
        if subtopic_id == OTHER_ID:
            return OTHER_TOPIC_ID
        return self.by_id[subtopic_id].topic_id


def validate_taxonomy(data: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["the file must contain a mapping at the top level"]
    if not _text(data.get("version")):
        problems.append("'version' must be a non-empty text")
    topics = data.get("topics")
    if not isinstance(topics, list) or not topics:
        problems.append("'topics' must be a non-empty list")
        topics = []
    seen_topics: set[str] = set()
    seen_sub: set[str] = set()
    for n, topic in enumerate(topics):
        if not isinstance(topic, dict) or not _text(topic.get("id")):
            problems.append(f"topics[{n}]: needs an 'id'")
            continue
        tid = topic["id"]
        if tid in seen_topics:
            problems.append(f"topic id '{tid}' is used twice")
        seen_topics.add(tid)
        if tid == OTHER_TOPIC_ID:
            problems.append(f"topic id '{OTHER_TOPIC_ID}' is reserved for the 'other' entry")
        for field_name in ("name", "description"):
            if not _text(topic.get(field_name)):
                problems.append(f"topic '{tid}': '{field_name}' is empty")
        subs = topic.get("subtopics")
        if not isinstance(subs, list) or not subs:
            problems.append(f"topic '{tid}': needs at least one subtopic")
            continue
        for m, sub in enumerate(subs):
            if not isinstance(sub, dict) or not _text(sub.get("id")):
                problems.append(f"topic '{tid}' subtopics[{m}]: needs an 'id'")
                continue
            sid = sub["id"]
            if sid in seen_sub:
                problems.append(f"subtopic id '{sid}' is used twice")
            seen_sub.add(sid)
            if not sid.startswith(tid + "."):
                problems.append(f"subtopic '{sid}' must start with its topic id '{tid}.'")
            for field_name in ("name", "description"):
                if not _text(sub.get(field_name)):
                    problems.append(f"subtopic '{sid}': '{field_name}' is empty")
            kw = sub.get("keywords", [])
            if not isinstance(kw, list) or any(not _text(k) for k in kw):
                problems.append(f"subtopic '{sid}': 'keywords' must be a list of text")
    other = data.get("other")
    if not isinstance(other, dict) or other.get("id") != OTHER_ID:
        problems.append(f"'other' must exist with id '{OTHER_ID}'")
    return problems


def load_taxonomy(path: Path | str) -> Taxonomy:
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InputError(f"taxonomy problems:\n- cannot read {path}: {exc}") from exc
    _raise("taxonomy", validate_taxonomy(data))
    subs = []
    for topic in data["topics"]:
        for sub in topic["subtopics"]:
            subs.append(Subtopic(id=sub["id"], name=sub["name"].strip(), topic_id=topic["id"],
                                 topic_name=topic["name"].strip(),
                                 description=" ".join(sub["description"].split()),
                                 keywords=tuple(sub.get("keywords", []))))
    tax = Taxonomy(version=data["version"], subtopics=tuple(subs), sha=file_sha(raw), path=path,
                   by_id={s.id: s for s in subs})
    return tax


# ------------------------------------------------------------------ cluster assignments

def load_assignments(path: Path | str, taxonomy: Taxonomy) -> dict[int, str]:
    """activity_id -> the subtopic id that clustering (step 4/5b) gave the post."""
    path = Path(path)
    problems: list[str] = []
    result: dict[int, str] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:      # utf-8-sig: Excel adds an invisible BOM
            reader = csv.DictReader(fh)
            missing = {"activity_id", "subtopic_id"} - set(reader.fieldnames or [])
            if missing:
                raise InputError(f"assignments problems:\n- missing columns {sorted(missing)}")
            valid = set(taxonomy.choice_ids)
            for row_no, row in enumerate(reader, start=2):
                try:
                    aid = int(row["activity_id"])
                except (TypeError, ValueError):
                    problems.append(f"line {row_no}: activity_id '{row['activity_id']}' is not a whole number")
                    continue
                sid = (row["subtopic_id"] or "").strip()
                if sid not in valid:
                    problems.append(f"line {row_no}: subtopic_id '{sid}' is not in the taxonomy")
                if aid in result:
                    problems.append(f"line {row_no}: activity_id {aid} appears twice")
                result[aid] = sid
    except OSError as exc:
        raise InputError(f"assignments problems:\n- cannot read {path}: {exc}") from exc
    _raise("assignments", problems)
    return result


# ------------------------------------------------------------------ posts

@dataclass(frozen=True)
class Post:
    activity_id: int
    text: str
    attachment_title: str = ""
    attachment_site: str = ""
    links: str = ""
    truncated: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def load_posts(path: Path | str) -> tuple[list[Post], list[tuple[int, str]]]:
    """Return (posts to classify, [(activity_id, reason) for rows that were skipped])."""
    import pandas as pd

    path = Path(path)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise InputError(f"posts problems:\n- cannot read {path}: {exc}") from exc
    problems: list[str] = []
    for col in ("activity_id", "text_clean"):
        if col not in df.columns:
            problems.append(f"missing column '{col}'")
    _raise("posts", problems)

    posts: list[Post] = []
    skipped: list[tuple[int, str]] = []
    seen: set[int] = set()
    for line_no, row in enumerate(df.to_dict("records"), start=2):
        try:
            aid = int(_clean(row["activity_id"]))
        except ValueError:
            problems.append(f"line {line_no}: activity_id '{row['activity_id']}' is not a whole number")
            continue
        if aid in seen:
            problems.append(f"line {line_no}: activity_id {aid} appears twice")
            continue
        seen.add(aid)
        text = _clean(row.get("text_clean")) or _clean(row.get("text"))
        if not text:
            skipped.append((aid, "no text"))
            continue
        truncated = len(text) > MAX_POST_CHARS
        posts.append(Post(activity_id=aid, text=text[:MAX_POST_CHARS],
                          attachment_title=_clean(row.get("attachment_title")),
                          attachment_site=_clean(row.get("attachment_site")),
                          links=_clean(row.get("urls"))[:300], truncated=truncated))
    _raise("posts", problems)
    return posts, skipped
