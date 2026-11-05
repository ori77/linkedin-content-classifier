"""
STEP 3 - Turn each post's keyphrases into an embedding (a list of 768 numbers).

Reads : keyphrases.jsonl   (made by step 2)
Writes: embeddings.npy        - the numbers, one row per post
        embeddings_meta.json  - which post each row belongs to + how it was made

Usage (PowerShell, venv active):
    pip install numpy requests
    ollama pull nomic-embed-text
    python step3_embeddings.py keyphrases.jsonl --limit 20     (quick trial)
    python step3_embeddings.py keyphrases.jsonl                (everything)
"""

import argparse
import json
import re
import sys
import time

import numpy as np
import requests

# sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text was trained with a short "task label" at the start of the text.
# "clustering: " tells it we want vectors that are good for grouping similar items.
TASK_PREFIX = "clustering: "

BATCH_SIZE = 32          # how many texts we send to Ollama in one request

# Matches a standalone year like 2026 or 1999 (but not "gpt4" or "100k").
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


# ----------------------------------------------------------------------------
# 1. READING THE KEYPHRASES
# ----------------------------------------------------------------------------
def load_keyphrases(path):
    """Return {activity_id: [phrases]} in file order. Broken lines are skipped.
    If a post appears twice, the later line wins."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                result[record["activity_id"]] = record["keyphrases"]
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def build_text(phrases):
    """Turn ['rag evaluation', '2026 travel trends'] into the text we embed."""
    cleaned = []
    for phrase in phrases:
        phrase = " ".join(YEAR_PATTERN.sub("", phrase).split())  # drop year, fix spaces
        if phrase:                                               # drop if nothing is left
            cleaned.append(phrase)
    return TASK_PREFIX + ", ".join(cleaned)


# ----------------------------------------------------------------------------
# 2. TALKING TO OLLAMA
# ----------------------------------------------------------------------------
def embed_batch(texts, model, url):
    """Send a list of texts, get back a (len(texts), 768) array of numbers."""
    try:
        response = requests.post(
            f"{url}/api/embed",
            json={"model": model, "input": texts, "keep_alive": "10m"},
            timeout=600,
        )
    except requests.exceptions.ConnectionError:
        sys.exit("Cannot reach Ollama. Is it running? Try: ollama ps")

    if response.status_code == 404:
        sys.exit(f"Ollama does not have '{model}'. Run: ollama pull {model}")
    response.raise_for_status()

    vectors = np.array(response.json()["embeddings"], dtype=np.float32)
    if vectors.shape[0] != len(texts):
        sys.exit(f"Asked for {len(texts)} embeddings, got {vectors.shape[0]}")
    return vectors


def normalise(vectors):
    """Scale every vector to length 1.

    With length-1 vectors, 'how similar are two posts' is just a dot product
    (this is called cosine similarity), and it ranges from -1 to 1.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(lengths, 1e-12, None)


# ----------------------------------------------------------------------------
# 3. A SANITY CHECK YOU CAN READ
# ----------------------------------------------------------------------------
def show_neighbours(ids, phrases_by_id, vectors, how_many=3, seed=42):
    """Pick a few random posts and print the 3 posts most similar to each."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ids), size=min(how_many, len(ids)), replace=False)

    print("\nSanity check - for a few random posts, their 3 closest neighbours:")
    for i in picks:
        similarity = vectors @ vectors[i]          # one dot product per post
        closest = np.argsort(-similarity)          # highest similarity first
        closest = [j for j in closest if j != i][:3]  # skip the post itself
        print(f"\n  POST: {', '.join(phrases_by_id[ids[i]])}")
        for j in closest:
            print(f"    {similarity[j]:.2f}  {', '.join(phrases_by_id[ids[j]])}")


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyphrases_path")
    parser.add_argument("--out-prefix", default="embeddings")
    parser.add_argument("--limit", type=int, default=None,
                        help="embed only the first N posts (for a quick trial)")
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    args = parser.parse_args()

    phrases_by_id = load_keyphrases(args.keyphrases_path)
    ids = list(phrases_by_id)
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        sys.exit("No keyphrases found. Has step 2 saved anything yet?")

    texts = [build_text(phrases_by_id[i]) for i in ids]
    print(f"Posts to embed: {len(ids)} | model: {args.model}")
    print(f"Example text sent to the model:\n  {texts[0]}\n")

    started = time.time()
    chunks = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunks.append(embed_batch(texts[start:start + BATCH_SIZE], args.model, args.url))
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    vectors = normalise(np.vstack(chunks))
    print(f"\nDone in {time.time() - started:.1f}s | shape = {vectors.shape} "
          f"(posts x dimensions)")

    np.save(f"{args.out_prefix}.npy", vectors)
    with open(f"{args.out_prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "task_prefix": TASK_PREFIX,
            "dimensions": int(vectors.shape[1]),
            "text_source": "keyphrases, years removed",
            "activity_ids": ids,
        }, f, ensure_ascii=False, indent=1)
    print(f"Saved {args.out_prefix}.npy and {args.out_prefix}_meta.json")

    show_neighbours(ids, phrases_by_id, vectors)


if __name__ == "__main__":
    main()
