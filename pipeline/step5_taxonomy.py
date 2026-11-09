"""
STEP 5 (v2) - Turn the clusters into a draft topic list (taxonomy) for YOU to review.

Reads : clusters_final.csv   (step 4)  + the embeddings it was made from (step 3)
Writes (into ./taxonomy/):
    taxonomy.draft.yaml        the draft topic list: topics > subtopics
    taxonomy_review.md         what to read and edit before approving it
    taxonomy_assignments.csv   which topic/subtopic every post landed in
    cluster_names.jsonl        saved answers: cluster names           (lets you resume)
    topic_proposals.jsonl      saved answers: proposed topic lists
    topic_assignments.jsonl    saved answers: which topic each cluster joined

Three steps:
    A. NAME each cluster            (one small LLM call per cluster)
    B. PROPOSE 7-11 parent topics   (ONE LLM call; you may use a bigger model: --topic-model)
    C. PLACE each cluster on a topic (one small LLM call per cluster, multiple-choice, so no
                                      cluster can be forgotten and no topic invented)
Plus, without any LLM: MEASURE how tight / how generic each cluster is.

Usage (PowerShell, venv active):
    python step5_taxonomy.py clusters_final.csv --emb-prefix embeddings_subject
    python step5_taxonomy.py clusters_final.csv --emb-prefix embeddings_subject --topic-model gemma3:12b --out-prefix taxonomy_gemma
    python step5_taxonomy.py clusters_final.csv --emb-prefix embeddings_subject --only 18,27,32
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date

import numpy as np
import yaml

from ollama_json import chat_json, LLMError, OllamaUnavailable, OLLAMA_URL
from step4_cluster import centre_similarities, top_words_per_cluster, words_of

DEFAULT_MODEL = "qwen3:latest"
PROMPT_VERSION = "taxonomy-v1"     # change this when a prompt changes, so old answers are not reused
CENTRAL_EXAMPLES = 8               # the most typical posts of a cluster
SPREAD_EXAMPLES = 4                # plus a few from further out, to judge how coherent it is
MIN_TOPICS, MAX_TOPICS = 8, 15       # the target for the FINAL list (a warning if outside)
PROPOSE_MIN, PROPOSE_MAX = 7, 11     # what we ask the model to propose (General is added by code)
GROUPING_VERSION = "taxonomy-v2-topics"
GENERAL_TOPIC = {"name": "General AI Engineering",
                 "description": "Broad AI engineering posts without one specific subject: "
                                "overviews, mixed fundamentals and general advice."}
UNASSIGNED = "Unassigned (needs review)"
COMMON_WORD_SHARE = 0.4              # a word in >= 40% of clusters counts as 'common'
GENERIC_FRACTION = 0.25              # flag the 25% most generic-looking clusters
COHERENCE_LEVELS = ["clear", "mixed", "generic"]

# ----------------------------------------------------------------------------
# THE PROMPTS AND ANSWER SHAPES
# ----------------------------------------------------------------------------
NAMING_SYSTEM = """You are helping build a topic list (taxonomy) for a personal library of
saved LinkedIn posts about AI, machine learning, software engineering and careers.

You are shown ONE cluster of similar posts: its top words and example posts, each
example written as 3 to 5 keyphrases. Return:
- name: 2 to 5 words naming the SUBJECT of the posts, in Title Case. Name the subject,
  not the format: do not use words like interview, roadmap, course, guide or tips
  unless the subject itself is interviewing or education.
- description: ONE sentence saying what posts in this cluster cover, precise enough
  that someone could decide whether a new post belongs here.
- coherence: "clear" if the examples share one specific subject; "mixed" if they cover
  two or more distinct subjects; "generic" if they are broad AI or engineering posts
  with no specific shared subject.
- reason: ONE short sentence justifying the coherence rating.
Use only the information shown."""

NAMING_SHAPE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "coherence": {"type": "string", "enum": COHERENCE_LEVELS},
        "reason": {"type": "string"},
    },
    "required": ["name", "description", "coherence", "reason"],
}

PROPOSE_SYSTEM = f"""You are designing the parent topics (a taxonomy) for a personal library of
saved LinkedIn posts about AI, machine learning, software engineering and careers.

You receive a list of CLUSTERS of posts, each with a name, a description and a size.
Propose between {PROPOSE_MIN} and {PROPOSE_MAX} parent TOPICS that together cover all clusters.

Rules:
- Each topic is ONE specific subject area (a technique, a technology, a discipline or an
  activity) that several related clusters can sit under.
- Topics must NOT overlap. Before answering, compare every pair of topics: if the clusters of
  one could also sit under the other, merge them. Group by subject, not by audience: all
  clusters about careers, jobs, hiring and salary belong in a single topic.
- Do NOT use broad catch-all names that describe everything in this library, such as
  "AI Engineering Practices", "AI Systems", "AI Development" or "AI Engineering". A topic name
  must say WHAT the posts are about. Words like "practices", "systems", "engineering" and
  "development" are fine inside a specific name ("LLM Inference Optimization") but not as the
  whole subject.
- Most topics should cover at least 2 clusters or about 30 posts; make a smaller topic only if
  its subject is truly distinct.
- Topic name: 2 to 5 words, Title Case, naming a SUBJECT (not a format such as interview,
  course or tips).
- Topic description: ONE sentence saying what belongs in the topic, precise enough that
  clusters can be sorted into it.
- Do NOT create a topic for broad, general posts; a topic called "{GENERAL_TOPIC['name']}"
  is added automatically for those."""

PROPOSE_SHAPE = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
            },
        }
    },
    "required": ["topics"],
}

ASSIGN_SYSTEM = f"""You are placing ONE cluster of saved LinkedIn posts onto exactly one TOPIC
from a fixed list.

Rules:
- Choose the MOST SPECIFIC topic whose description matches the SUBJECT of the cluster. A broad
  topic is right only when no specific topic fits.
- Use "{GENERAL_TOPIC['name']}" only when the cluster is broad and fits no specific topic. A
  cluster with a clearly specific subject (for example MLOps, tokenization, data structures)
  belongs on the topic for that subject, even if some of its wording is general.
- reason: ONE sentence naming the concrete subject of the cluster that matches the topic. Do not
  just say that it "aligns with" the topic."""


def assign_shape(topic_names):
    """The answer must be one of the topic names: the model cannot invent one."""
    return {
        "type": "object",
        "properties": {"topic": {"type": "string", "enum": list(topic_names)},
                       "reason": {"type": "string"}},
        "required": ["topic", "reason"],
    }


# ----------------------------------------------------------------------------
# 1. LOADING THE CLUSTERS AND PREPARING WHAT THE LLM WILL SEE
# ----------------------------------------------------------------------------
def load_clusters(csv_path, emb_prefix):
    """Returns ids, labels (array), seen phrases per post (list of lists), vectors."""
    vectors = np.load(f"{emb_prefix}.npy")
    with open(f"{emb_prefix}_meta.json", encoding="utf-8") as f:
        meta_ids = json.load(f)["activity_ids"]

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r["activity_id"] for r in rows]
    if ids != meta_ids:
        sys.exit("The clusters file and the embeddings are not about the same posts in the "
                 "same order. Use the --emb-prefix that step 4 was run with.")
    labels = np.array([int(r["cluster"]) for r in rows])
    seen = [[p for p in r["keyphrases_used"].split("; ") if p] for r in rows]
    return ids, labels, seen, vectors


def pick_examples(ranked, central=CENTRAL_EXAMPLES, spread=SPREAD_EXAMPLES):
    """From posts ranked 'most typical first', take the top `central`, plus `spread`
    evenly spaced ones from the rest, so the LLM also sees the edges of the cluster."""
    chosen = list(ranked[:central])
    rest = list(ranked[central:])
    if rest and spread:
        positions = np.linspace(0, len(rest) - 1, num=min(spread, len(rest)))
        for p in sorted(set(int(round(x)) for x in positions)):
            chosen.append(rest[p])
    return chosen


def build_cards(labels, seen, vectors):
    """For each cluster: size, top words, and example posts (as the LLM will see them)."""
    top_words = top_words_per_cluster(labels, [words_of(p) for p in seen])
    cards = {}
    for c in sorted(set(labels.tolist()) - {-1}):
        rows = np.where(labels == c)[0]
        order = rows[np.argsort(-centre_similarities(vectors, rows))]
        examples = [seen[r] for r in pick_examples(list(order))]
        cards[c] = {"size": int(len(rows)), "top_words": top_words[c], "examples": examples}
    return cards


def naming_prompt(card):
    lines = [f"Cluster size: {card['size']} posts",
             f"Top words: {', '.join(card['top_words'])}",
             f"Example posts (1-{CENTRAL_EXAMPLES} are the most typical; the rest are more peripheral):"]
    for i, phrases in enumerate(card["examples"], start=1):
        lines.append(f"{i}. {'; '.join(phrases)}")
    return "\n".join(lines)


def proposal_prompt(named):
    lines = [f"{len(named)} clusters:"]
    for c in sorted(named):
        n = named[c]
        lines.append(f"- {n['name']} | {n['description']} | {n['size']} posts")
    return "\n".join(lines)


def assignment_prompt(topics, card, cluster):
    """What the model sees when placing one cluster. `cluster` is its named dict."""
    lines = ["Topics:"]
    lines += [f"- {t['name']}: {t['description']}" for t in topics]
    lines += ["", "Cluster to place:",
              f"Name: {cluster['name']}",
              f"Description: {cluster['description']}",
              f"Size: {cluster['size']} posts",
              f"Top words: {', '.join(card['top_words'])}"]
    if cluster.get("looks_generic"):
        lines.append("Measured hint: much of this cluster's wording is broad and shared with many "
                     "other clusters. This is only a hint: if its subject is clearly specific, "
                     "place it on the specific topic.")
    lines.append("Typical posts:")
    lines += [f"{i}. {'; '.join(p)}" for i, p in enumerate(card["examples"][:3], start=1)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 2. SAVING ANSWERS, SO A STOPPED RUN CAN CONTINUE
# ----------------------------------------------------------------------------
def cache_key(*parts):
    """A short fingerprint of everything that decides the answer. If any part changes
    (the prompt, the model, thinking on/off, the examples), the old answer is not reused."""
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def load_name_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[(rec["cluster"], rec["key"])] = rec
                except (json.JSONDecodeError, KeyError):
                    continue          # a half-written last line is simply ignored
    return cache


def append_jsonl(path, record):
    needs_newline = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# 3. TASK A: NAME EVERY CLUSTER
# ----------------------------------------------------------------------------
def tidy_name_answer(answer):
    """Check and tidy the model's answer; raise LLMError if it is unusable."""
    name = " ".join(str(answer.get("name", "")).split()).strip(" .\"'")
    description = " ".join(str(answer.get("description", "")).split())
    if not name or len(name) > 60:
        raise LLMError(f"unusable name: {name!r}")
    if not description:
        raise LLMError("empty description")
    coherence = answer.get("coherence")
    if coherence not in COHERENCE_LEVELS:        # never trust the model with the label set
        coherence = "mixed"
    return {"name": name, "description": description,
            "coherence": coherence,
            "reason": " ".join(str(answer.get("reason", "")).split())}


def name_cluster(card, args, attempts=2):
    """Ask the model to name one cluster. Retries once on an unusable answer."""
    last = None
    for _ in range(attempts):
        try:
            answer, stats = chat_json(
                args.model, NAMING_SYSTEM, naming_prompt(card), NAMING_SHAPE, url=args.url,
                num_ctx=args.num_ctx, think=args.think, num_predict=args.max_tokens)
            return tidy_name_answer(answer), stats, None
        except OllamaUnavailable as e:
            sys.exit(str(e))
        except LLMError as e:
            last = str(e)
    return None, None, last


def name_all_clusters(cards, wanted, args, out_dir):
    path = os.path.join(out_dir, "cluster_names.jsonl")
    cache = load_name_cache(path)
    named, durations = {}, []

    for i, c in enumerate(wanted, start=1):
        card = cards[c]
        key = cache_key(PROMPT_VERSION, args.model, args.think, NAMING_SYSTEM,
                        naming_prompt(card))
        if (c, key) in cache:
            rec = cache[(c, key)]
            named[c] = {**rec["result"], "size": card["size"]}
            print(f"[{i}/{len(wanted)}] cluster {c:>3} ({card['size']:>3} posts) | saved earlier | "
                  f"{named[c]['name']} [{named[c]['coherence']}]")
            continue

        result, stats, error = name_cluster(card, args)
        if result is None or stats is None:
            print(f"[{i}/{len(wanted)}] cluster {c:>3} FAILED ({error}) - using a placeholder name")
            named[c] = {"name": f"Cluster {c}", "description": "Naming failed; please name by hand.",
                        "coherence": "mixed", "reason": "naming failed", "size": card["size"]}
            continue

        append_jsonl(path, {"cluster": c, "key": key, "result": result, "model": args.model,
                            "think": args.think, "seconds": stats["seconds"]})
        named[c] = {**result, "size": card["size"]}
        durations.append(stats["seconds"])
        eta = (sum(durations) / len(durations)) * (len(wanted) - i)
        print(f"[{i}/{len(wanted)}] cluster {c:>3} ({card['size']:>3} posts) | {stats['seconds']:>5.1f}s | "
              f"ETA {int(eta // 60)} min | {result['name']} [{result['coherence']}]")
    return named


# ----------------------------------------------------------------------------
# 3b. MEASURING HOW GENERIC EACH CLUSTER IS (no LLM involved)
# ----------------------------------------------------------------------------
def disambiguate_names(named, cards):
    """Clusters that got the same name get a tag from their top words, so they can be told
    apart: 'AI Engineering Practices (langchain, langgraph)'."""
    by_name = {}
    for c, n in named.items():
        by_name.setdefault(n["name"].lower(), []).append(c)
    taken = {n["name"].lower() for n in named.values()}
    for same in by_name.values():
        if len(same) < 2:
            continue
        for c in sorted(same):
            base = named[c]["name"]
            in_name = {w.strip("()").lower() for w in base.split()}
            extra = [w for w in cards[c]["top_words"] if w not in in_name][:2]
            new = f"{base} ({', '.join(extra)})" if extra else base
            if new.lower() in taken - {base.lower()}:
                new = f"{base} #{c}"
            taken.add(new.lower())
            named[c]["name"] = new
    return named


def cluster_metrics(labels, seen, vectors):
    """Two numbers per cluster, no LLM needed:
      generic_share how much of its wording consists of words that appear in many other
                    clusters too (high = broad, 'ai engineer / llm / rag' style posts).
                    THIS drives the 'looks generic' flag.
      tightness     how close its posts are to their centre. Shown for information only:
                    it does not separate generic from specific clusters (see flag_generic)."""
    clusters = sorted(set(labels.tolist()) - {-1})
    words_in = {c: set() for c in clusters}
    per_post = {c: [] for c in clusters}
    for label, phrases in zip(labels.tolist(), seen):
        if label == -1:
            continue
        words = words_of(phrases)
        words_in[label].update(words)
        per_post[label].append(words)
    appears_in = Counter(w for c in clusters for w in words_in[c])
    common = {w for w, k in appears_in.items() if k >= COMMON_WORD_SHARE * len(clusters)}

    metrics = {}
    for c in clusters:
        rows = np.where(labels == c)[0]
        shares = [sum(w in common for w in ws) / len(ws) for ws in per_post[c] if ws]
        metrics[c] = {"tightness": float(centre_similarities(vectors, rows).mean()),
                      "generic_share": float(np.mean(shares)) if shares else 0.0}
    return metrics


def flag_generic(metrics, fraction=GENERIC_FRACTION):
    """The `fraction` of clusters whose wording is the most generic (highest generic share).

    Tightness is deliberately NOT used: on real data the generic clusters turned out to be the
    TIGHTEST ones (keyphrases like 'ai; llm fundamentals' embed almost identically), and big
    specific clusters the loosest. Tightness measures similarity, not specificity."""
    ids = sorted(metrics)
    if len(ids) < 2:
        return set()
    how_many = max(1, round(fraction * len(ids)))
    worst_first = sorted(ids, key=lambda c: -metrics[c]["generic_share"])
    return set(worst_first[:how_many])


# ----------------------------------------------------------------------------
# 4. STEP B: PROPOSE THE PARENT TOPICS  /  STEP C: PLACE EACH CLUSTER
# ----------------------------------------------------------------------------
def validate_topics(raw):
    """Clean the model's proposed topics. Returns (topics, problems)."""
    seen_names, topics = set(), []
    for t in raw.get("topics", []):
        name = " ".join(str(t.get("name", "")).split()).strip(" .\"'")
        description = " ".join(str(t.get("description", "")).split())
        key = name.lower()
        if (not name or len(name) > 60 or not description
                or key in seen_names or key == GENERAL_TOPIC["name"].lower()):
            continue
        seen_names.add(key)
        topics.append({"name": name, "description": description})
    problems = []
    if not PROPOSE_MIN <= len(topics) <= PROPOSE_MAX:
        problems.append(f"You proposed {len(topics)} usable topics; I need between "
                        f"{PROPOSE_MIN} and {PROPOSE_MAX}.")
    return topics, problems


def propose_topics(named, args, out_dir):
    """Ask the model for the parent topics (one call, one retry that says what was wrong)."""
    model = args.topic_model or args.model
    path = os.path.join(out_dir, "topic_proposals.jsonl")
    prompt = proposal_prompt(named)
    key = cache_key(GROUPING_VERSION, model, args.think, PROPOSE_SYSTEM, prompt)
    cache = load_name_cache(path)               # reused: records use cluster = -1
    if (-1, key) in cache:
        topics = cache[(-1, key)]["result"]["topics"]
        print(f"Topic list: using the answer saved earlier ({len(topics)} topics, model {model})")
        return topics

    user, topics, problems = prompt, [], ["no answer"]
    for attempt in (1, 2):
        try:
            raw, stats = chat_json(model, PROPOSE_SYSTEM, user, PROPOSE_SHAPE, url=args.url,
                                   num_ctx=args.num_ctx, think=args.think,
                                   num_predict=args.max_tokens_grouping)
        except OllamaUnavailable as e:
            sys.exit(str(e))
        except LLMError as e:
            problems = [f"Your last answer was unusable ({e}). Answer again."]
            user = prompt + "\n\n" + problems[0]
            continue
        found, problems = validate_topics(raw)
        if found:
            topics = found
        print(f"Topic list attempt {attempt} ({model}): {len(found)} topics, {stats['seconds']}s"
              + (f" | problems: {' '.join(problems)}" if problems else ""))
        if not problems:
            break
        user = prompt + "\n\nYour previous answer had problems: " + " ".join(problems) + \
            " Fix them and answer again."
    if not topics:
        sys.exit("The model did not produce a usable topic list. Try --topic-model or --think.")
    append_jsonl(path, {"cluster": -1, "key": key, "result": {"topics": topics},
                        "model": model, "think": args.think})
    return topics


def place_cluster(prompt, topic_names, args, attempts=2):
    """Ask the model which topic one cluster belongs to. Returns (answer, error)."""
    last = None
    for _ in range(attempts):
        try:
            answer, _stats = chat_json(
                args.model, ASSIGN_SYSTEM, prompt, assign_shape(topic_names), url=args.url,
                num_ctx=args.num_ctx, think=args.think, num_predict=args.max_tokens)
            if answer.get("topic") in topic_names:
                return {"topic": answer["topic"],
                        "reason": " ".join(str(answer.get("reason", "")).split())}, None
            last = f"answered with an unknown topic: {answer.get('topic')!r}"
        except OllamaUnavailable as e:
            sys.exit(str(e))
        except LLMError as e:
            last = str(e)
    return None, last


def place_all_clusters(named, cards, topics, args, out_dir):
    """Returns (topics_with_clusters, warnings). Every cluster ends up on exactly one topic."""
    path = os.path.join(out_dir, "topic_assignments.jsonl")
    cache = load_name_cache(path)
    options = topics + [GENERAL_TOPIC]
    topic_names = [t["name"] for t in options]
    members = {name: [] for name in topic_names}
    members[UNASSIGNED] = []
    reasons, warnings_, durations = {}, [], []
    ordered = sorted(named)

    for i, c in enumerate(ordered, start=1):
        prompt = assignment_prompt(options, cards[c], named[c])
        key = cache_key(GROUPING_VERSION, args.model, args.think, ASSIGN_SYSTEM, prompt)
        if (c, key) in cache:
            result, note = cache[(c, key)]["result"], "saved earlier"
        else:
            started = time.time()
            result, error = place_cluster(prompt, topic_names, args)
            if result is None:
                print(f"[{i}/{len(ordered)}] cluster {c:>3} FAILED ({error}) -> {UNASSIGNED}")
                members[UNASSIGNED].append(c)
                warnings_.append(f"Cluster {c} ({named[c]['name']}) could not be placed: {error}")
                continue
            append_jsonl(path, {"cluster": c, "key": key, "result": result, "model": args.model,
                                "think": args.think})
            durations.append(time.time() - started)
            eta = (sum(durations) / len(durations)) * (len(ordered) - i)
            note = f"{durations[-1]:>4.1f}s | ETA {int(eta // 60)} min"
        members[result["topic"]].append(c)
        reasons[c] = result["reason"]
        print(f"[{i}/{len(ordered)}] cluster {c:>3} | {note} | {named[c]['name']}  ->  {result['topic']}")

    result_topics = []
    for t in options + [{"name": UNASSIGNED,
                         "description": "Clusters the model could not place; decide by hand."}]:
        if members[t["name"]]:
            result_topics.append({**t, "cluster_ids": members[t["name"]]})
    if not MIN_TOPICS <= len([t for t in result_topics if t["name"] != UNASSIGNED]) <= MAX_TOPICS:
        warnings_.append(f"The final list has {len(result_topics)} topics, outside the "
                         f"{MIN_TOPICS}-{MAX_TOPICS} target.")
    return result_topics, warnings_, reasons


# ----------------------------------------------------------------------------
# 5. BUILDING AND WRITING THE RESULTS
# ----------------------------------------------------------------------------
def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def unique_slug(base, used):
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def build_taxonomy(topics, named, cards, noise_count):
    """Combine the topics and the cluster names into the taxonomy structure."""
    used_topic, result = set(), []
    for t in topics:
        topic_id = unique_slug(slugify(t["name"]), used_topic)
        used_sub, subtopics = set(), []
        for c in sorted(t["cluster_ids"], key=lambda c: -named[c]["size"]):
            n = named[c]
            subtopics.append({
                "id": f"{topic_id}.{unique_slug(slugify(n['name']), used_sub)}",
                "name": n["name"],
                "description": n["description"],
                "cluster": c,
                "post_count": n["size"],
                "looks_generic": bool(n.get("looks_generic", False)),
                "tightness": round(n.get("tightness", 0.0), 3),
                "generic_share": round(n.get("generic_share", 0.0), 3),
                "model_coherence": n["coherence"],
                "keywords": cards[c]["top_words"][:6],
            })
        result.append({"id": topic_id, "name": t["name"], "description": t["description"],
                       "post_count": sum(s["post_count"] for s in subtopics),
                       "subtopics": subtopics})
    result.sort(key=lambda t: (t["name"] == UNASSIGNED, -t["post_count"]))
    return {"topics": result,
            "other": {"id": "other.general", "name": "Other / General",
                      "description": "Posts that fit no topic clearly.",
                      "post_count": noise_count}}


def write_yaml(path, taxonomy, args):
    doc = {"version": "draft",
           "created": date.today().isoformat(),
           "made_by": {"model": args.model, "topic_model": args.topic_model or args.model,
                       "thinking": args.think, "prompt": PROMPT_VERSION,
                       "grouping": GROUPING_VERSION},
           **taxonomy}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=100)


def write_assignments(path, ids, labels, taxonomy):
    by_cluster = {s["cluster"]: (t["id"], s["id"])
                  for t in taxonomy["topics"] for s in t["subtopics"]}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["activity_id", "cluster", "topic_id", "subtopic_id"])
        for post_id, label in zip(ids, labels):
            topic_id, sub_id = by_cluster.get(int(label), ("other", taxonomy["other"]["id"]))
            writer.writerow([post_id, int(label), topic_id, sub_id])


def write_review(path, taxonomy, named, cards, noise_examples, warnings_, reasons):
    total = sum(t["post_count"] for t in taxonomy["topics"]) + taxonomy["other"]["post_count"]
    subs = [s for t in taxonomy["topics"] for s in t["subtopics"]]
    lines = ["# Draft topic list: please review", "",
             f"**{len(taxonomy['topics'])} topics** covering **{total} posts** "
             f"({taxonomy['other']['post_count']} in Other / General).", ""]
    for w in warnings_:
        lines.append(f"> ⚠️ {w}")
    if warnings_:
        lines.append("")
    lines += ["**How to review:** read each topic. In `taxonomy.draft.yaml`, rename things, "
              "move a subtopic under another topic, merge two subtopics (delete one and keep "
              "the better description), or delete a subtopic that is not a real subject.", "",
              "**Markers:** 🔴 looks generic (measured: much of its wording is made of words shared "
              "with many other clusters), 🟡 the model itself called it mixed, 🟢 neither. "
              "Tightness is shown for information only.", "",
              "## Most generic-looking clusters (measured)", "",
              "| cluster | name | posts | tightness | generic share | topic |", "|---|---|---|---|---|---|"]
    topic_of = {s["cluster"]: t["name"] for t in taxonomy["topics"] for s in t["subtopics"]}
    for s in sorted([s for s in subs if s["looks_generic"]], key=lambda s: -s["generic_share"]):
        lines.append(f"| {s['cluster']} | {s['name']} | {s['post_count']} | {s['tightness']:.2f} "
                     f"| {s['generic_share']:.0%} | {topic_of[s['cluster']]} |")
    lines.append("")
    for t in taxonomy["topics"]:
        lines += [f"## {t['name']} ({t['post_count']} posts)", f"*{t['description']}*", ""]
        for s in t["subtopics"]:
            icon = "🔴" if s["looks_generic"] else ("🟡" if s["model_coherence"] != "clear" else "🟢")
            lines.append(f"### {icon} {s['name']} ({s['post_count']} posts, cluster {s['cluster']})")
            lines.append(f"{s['description']}")
            lines.append(f"- Why it is on this topic: {reasons.get(s['cluster'], 'n/a')}")
            lines.append(f"- Tightness {s['tightness']:.2f}, generic share {s['generic_share']:.0%}")
            lines.append(f"- Keywords: {', '.join(s['keywords'])}")
            for phrases in cards[s["cluster"]]["examples"][:3]:
                lines.append(f"  - e.g. {'; '.join(phrases)}")
            lines.append("")
    lines += [f"## Other / General ({taxonomy['other']['post_count']} posts)",
              "Posts that did not belong to any cluster. A few examples:"]
    lines += [f"- {'; '.join(p)}" for p in noise_examples]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clusters_csv")
    parser.add_argument("--emb-prefix", default="embeddings")
    parser.add_argument("--out-dir", default="taxonomy")
    parser.add_argument("--out-prefix", default="taxonomy",
                        help="names of the result files; change it to keep two versions side by side")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--topic-model", default=None,
                        help="a different (maybe bigger) model just for proposing the topic list")
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--think", action="store_true",
                        help="let the model reason before answering (slower; was not better in our trial)")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="cap on the length of one naming/placing answer")
    parser.add_argument("--max-tokens-grouping", type=int, default=None,
                        help="cap on the length of the topic-list answer")
    parser.add_argument("--only", default="",
                        help="comma-separated cluster ids: only name these, then stop (a trial)")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 6000 if args.think else 600
    if args.max_tokens_grouping is None:
        args.max_tokens_grouping = 8000 if args.think else 2500

    os.makedirs(args.out_dir, exist_ok=True)
    ids, labels, seen, vectors = load_clusters(args.clusters_csv, args.emb_prefix)
    cards = build_cards(labels, seen, vectors)
    noise_count = int((labels == -1).sum())
    print(f"{len(ids)} posts | {len(cards)} clusters | {noise_count} noise | model {args.model}"
          f" | thinking {'on' if args.think else 'off'}")

    wanted = sorted(cards)
    if args.only:
        wanted = [int(x) for x in args.only.split(",") if x.strip()]
        unknown = [c for c in wanted if c not in cards]
        if unknown:
            sys.exit(f"No such clusters: {unknown}")

    try:
        named = name_all_clusters(cards, wanted, args, args.out_dir)
    except KeyboardInterrupt:
        sys.exit("\nStopped by you. Answers so far are saved - run the same command to continue.")

    if args.only:
        print("\nTrial finished (no topic list with --only). Run without --only for the full thing.")
        return

    metrics = cluster_metrics(labels, seen, vectors)
    generic = flag_generic(metrics)
    for c in named:
        named[c].update(metrics[c], looks_generic=c in generic)
    named = disambiguate_names(named, cards)
    print(f"\nLooks generic (measured): clusters {sorted(generic)}")

    print("\nProposing the parent topics...")
    try:
        topics = propose_topics(named, args, args.out_dir)
        print("Topics proposed: " + " | ".join(t["name"] for t in topics))
        print("\nPlacing each cluster on a topic...")
        topics, warnings_, reasons = place_all_clusters(named, cards, topics, args, args.out_dir)
    except KeyboardInterrupt:
        sys.exit("\nStopped by you. Answers so far are saved - run the same command to continue.")

    taxonomy = build_taxonomy(topics, named, cards, noise_count)
    noise_examples = [seen[r] for r in np.where(labels == -1)[0][:10]]
    p = args.out_prefix
    write_yaml(os.path.join(args.out_dir, f"{p}.draft.yaml"), taxonomy, args)
    write_assignments(os.path.join(args.out_dir, f"{p}_assignments.csv"), ids, labels, taxonomy)
    write_review(os.path.join(args.out_dir, f"{p}_review.md"), taxonomy, named, cards,
                 noise_examples, warnings_, reasons)

    print(f"\n{len(taxonomy['topics'])} topics:")
    for t in taxonomy["topics"]:
        print(f"  {t['post_count']:>4} posts | {t['name']} ({len(t['subtopics'])} subtopics)")
    for w in warnings_:
        print(f"WARNING: {w}")
    print(f"Saved into {args.out_dir}/: {p}.draft.yaml, {p}_review.md, {p}_assignments.csv")


if __name__ == "__main__":
    main()