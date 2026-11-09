"""
STEP 5b - Freeze the taxonomy: check YOUR edited draft and write taxonomy v1.

Reads : the draft you edited by hand (for example taxonomy/taxonomy_gemma.draft.yaml)
        clusters_final.csv + the embeddings it came from (to know every cluster and its size)
Writes (into ./taxonomy/):
    taxonomy.v1.yaml            the frozen taxonomy (this one is committed to git)
    taxonomy_assignments.v1.csv which topic/subtopic every post landed in (not committed)

What you may do to the draft by hand:
  * rename topics or subtopics, change descriptions
  * move a subtopic under another topic (cut and paste the whole block)
  * add a new topic (just name, description and a list of subtopics: no id or counts needed)
  * merge clusters into one subtopic:   clusters: [5, 7]      (instead of   cluster: 5)
  * send clusters to Other / General:   under `other:` add    clusters: [12, 30]
The script recomputes ids and post counts itself, and refuses to continue if anything is
inconsistent (a cluster missing or used twice, an empty topic, duplicate names ...).

Usage (PowerShell, venv active):
    python step5b_finalize.py taxonomy/taxonomy_gemma.draft.yaml clusters_final.csv --emb-prefix embeddings_subject
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import date

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from step5_taxonomy import (load_clusters, build_cards, slugify, unique_slug,
                            MIN_TOPICS, MAX_TOPICS, UNASSIGNED)


def tidy(text):
    return " ".join(str(text or "").split())


def clusters_of(sub):
    """The cluster ids of a subtopic: `cluster: 5` or `clusters: [5, 7]`. Returns (ids, problem)."""
    raw = sub.get("clusters", sub.get("cluster"))
    if raw is None:
        return [], "has no `cluster` or `clusters`"
    if not isinstance(raw, list):
        raw = [raw]
    if not raw or any(isinstance(c, bool) or not isinstance(c, int) for c in raw):
        return [], f"has unusable cluster ids: {raw!r}"
    return list(raw), None


def finalize(doc, sizes, top_words, noise_count):
    """Check the edited draft and build the frozen taxonomy.
    Returns (taxonomy, problems, warnings). Problems must be fixed; warnings are only shown."""
    problems, warnings_ = [], []
    if not isinstance(doc, dict) or not isinstance(doc.get("topics"), list):
        return None, ["The file has no `topics:` list. Is this the draft yaml?"], []

    used_topic_ids, topic_names, sub_names = set(), set(), set()
    owner = {}                                    # cluster id -> where it was first used
    topics = []
    for t in doc["topics"]:
        name = tidy(t.get("name") if isinstance(t, dict) else "")
        if not name:
            problems.append("A topic has no name.")
            continue
        label = f"Topic '{name}'"
        if name == UNASSIGNED:
            problems.append(f"{label} must be resolved: move its subtopics onto real topics.")
        if name.lower() in topic_names:
            problems.append(f"Two topics are called '{name}'.")
        topic_names.add(name.lower())
        if not tidy(t.get("description")):
            problems.append(f"{label} has no description.")
        topic_id = unique_slug(slugify(name), used_topic_ids)

        used_sub_ids, subtopics = set(), []
        for s in t.get("subtopics") or []:
            sname = tidy(s.get("name"))
            if not sname:
                problems.append(f"{label} has a subtopic without a name.")
                continue
            if sname.lower() in sub_names:
                problems.append(f"Two subtopics are called '{sname}'.")
            sub_names.add(sname.lower())
            if not tidy(s.get("description")):
                problems.append(f"Subtopic '{sname}' has no description.")
            ids, why = clusters_of(s)
            if why:
                problems.append(f"Subtopic '{sname}' {why}.")
            for c in ids:
                if c not in sizes:
                    problems.append(f"Subtopic '{sname}' uses cluster {c}, which does not exist.")
                elif c in owner:
                    problems.append(f"Cluster {c} is used twice: in '{owner[c]}' and in '{sname}'.")
                else:
                    owner[c] = sname
            valid = [c for c in ids if c in sizes]
            keywords = s.get("keywords") or (top_words.get(valid[0], [])[:6] if valid else [])
            subtopics.append({
                "id": f"{topic_id}.{unique_slug(slugify(sname), used_sub_ids)}",
                "name": sname,
                "description": tidy(s.get("description")),
                "clusters": sorted(valid, key=lambda c: -sizes[c]),
                "post_count": sum(sizes[c] for c in valid),
                "keywords": list(keywords),
            })
        if not subtopics:
            problems.append(f"{label} has no subtopics.")
        subtopics.sort(key=lambda s: -s["post_count"])
        topics.append({"id": topic_id, "name": name, "description": tidy(t.get("description")),
                       "post_count": sum(s["post_count"] for s in subtopics),
                       "subtopics": subtopics})

    other = doc.get("other") or {}
    moved = other.get("clusters") or []
    for c in moved:
        if c not in sizes:
            problems.append(f"`other` lists cluster {c}, which does not exist.")
        elif c in owner:
            problems.append(f"Cluster {c} is used twice: in '{owner[c]}' and in Other / General.")
        else:
            owner[c] = "Other / General"
    missing = sorted(set(sizes) - set(owner))
    if missing:
        problems.append(f"These clusters are not in any subtopic and not in `other`: {missing}.")

    if not MIN_TOPICS <= len(topics) <= MAX_TOPICS:
        warnings_.append(f"{len(topics)} topics (the usual target is {MIN_TOPICS}-{MAX_TOPICS}).")
    singles = [t["name"] for t in topics if len(t["subtopics"]) == 1]
    if singles:
        warnings_.append(f"Topics with a single subtopic: {', '.join(singles)}.")

    topics.sort(key=lambda t: -t["post_count"])
    moved_ok = sorted(c for c in moved if c in sizes and owner.get(c) == "Other / General")
    taxonomy = {"topics": topics,
                "other": {"id": "other.general", "name": "Other / General",
                          "description": tidy(other.get("description")) or
                          "Posts that fit no topic clearly.",
                          "post_count": noise_count + sum(sizes[c] for c in moved_ok),
                          "clusters": moved_ok}}
    return taxonomy, problems, warnings_


def write_assignments(path, ids, labels, taxonomy):
    where = {c: (t["id"], s["id"]) for t in taxonomy["topics"] for s in t["subtopics"]
             for c in s["clusters"]}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["activity_id", "cluster", "topic_id", "subtopic_id"])
        for post_id, label in zip(ids, labels):
            topic_id, sub_id = where.get(int(label), ("other", taxonomy["other"]["id"]))
            writer.writerow([post_id, int(label), topic_id, sub_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edited_yaml")
    parser.add_argument("clusters_csv")
    parser.add_argument("--emb-prefix", default="embeddings")
    parser.add_argument("--out-dir", default="taxonomy")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen taxonomy (it is meant to stay fixed)")
    args = parser.parse_args()

    out_yaml = os.path.join(args.out_dir, f"taxonomy.{args.version}.yaml")
    out_csv = os.path.join(args.out_dir, f"taxonomy_assignments.{args.version}.csv")
    if os.path.exists(out_yaml) and not args.force:
        sys.exit(f"{out_yaml} already exists. A frozen taxonomy should not change by accident; "
                 f"use --version v2 for a new one, or --force to overwrite.")

    with open(args.edited_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ids, labels, seen, vectors = load_clusters(args.clusters_csv, args.emb_prefix)
    cards = build_cards(labels, seen, vectors)
    sizes = {c: card["size"] for c, card in cards.items()}
    top_words = {c: card["top_words"] for c, card in cards.items()}
    noise_count = int((labels == -1).sum())

    taxonomy, problems, warnings_ = finalize(doc, sizes, top_words, noise_count)
    for w in warnings_:
        print(f"Note: {w}")
    if problems:
        print(f"\n{len(problems)} problem(s) to fix in {args.edited_yaml}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    frozen = {"version": args.version, "created": date.today().isoformat(),
              "based_on": os.path.basename(args.edited_yaml), **taxonomy}
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen, f, sort_keys=False, allow_unicode=True, width=100)
    write_assignments(out_csv, ids, labels, taxonomy)

    total = sum(t["post_count"] for t in taxonomy["topics"]) + taxonomy["other"]["post_count"]
    print(f"\nTaxonomy {args.version}: {len(taxonomy['topics'])} topics, "
          f"{sum(len(t['subtopics']) for t in taxonomy['topics'])} subtopics, {total} posts")
    for t in taxonomy["topics"]:
        print(f"  {t['post_count']:>4} posts | {t['name']} ({len(t['subtopics'])} subtopics)")
    print(f"  {taxonomy['other']['post_count']:>4} posts | Other / General")
    print(f"Saved: {out_yaml}\n       {out_csv}")


if __name__ == "__main__":
    main()
