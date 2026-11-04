"""
STEP 2 - Extract 3-5 keyphrases for every post, in a loop, saving after each one.

Usage (PowerShell, from your pipeline folder, venv active):
    python step2_keyphrases.py "..\\input-data\\posts_with_save_dates.csv" --limit 10
    python step2_keyphrases.py "..\\input-data\\posts_with_save_dates.csv"

Safe to stop (Ctrl+C) and re-run at any time: finished posts are skipped.
"""

import argparse
import csv
import json
import os
import sys
import time

import requests

# Windows CMD/PowerShell can crash when printing emojis or styled letters.
# sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# SETTINGS (same ideas as Step 1)
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:latest"
NUM_CTX = 4096          # the model's "working memory" in tokens
MAX_CHARS = 3500        # cut very long posts so they always fit in NUM_CTX
DEFAULT_OUT = "keyphrases.jsonl"

# The ONLY shape of answer we accept: an object with a list of 3-5 strings.
ANSWER_SHAPE = {
    "type": "object",
    "properties": {
        "keyphrases": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5,
        }
    },
    "required": ["keyphrases"],
}

SYSTEM_PROMPT = """You label LinkedIn posts for a topic-discovery system.
Return 3 to 5 keyphrases that capture WHAT THE POST IS ABOUT.

Rules:
- Each keyphrase is 1 to 4 words, lowercase, concrete and reusable
  (good: "rag evaluation", "product manager hiring", "python decorators").
- Describe the subject matter, not the writing style
  (bad: "interesting post", "must read", "great insights").
- Do not use names of people or the author's company unless the post is
  specifically about that company or tool.
- Each keyphrase must cover a DIFFERENT aspect of the post. Never give two
  phrases that are rewordings of each other.
- Do not include years or dates (no "2026").
- Write every word in full, with spaces between words.
- Put the most important keyphrase first.
- Use only information in the post."""


# ----------------------------------------------------------------------------
# 1. LOADING POSTS
# ----------------------------------------------------------------------------
def load_posts(csv_path):
    """Read the CSV and return a list of rows (dicts) that have usable text."""
    posts = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("activity_id") and row.get("text_clean", "").strip():
                posts.append(row)
    return posts


# ----------------------------------------------------------------------------
# 2. RESUME SUPPORT
# ----------------------------------------------------------------------------
def load_done_ids(out_path):
    """Return the set of activity_ids already saved in the output file.

    A line that cannot be parsed (e.g. the laptop died mid-write) is simply
    ignored - that post will be redone.
    """
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["activity_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def save_result(out_path, record):
    """Append ONE json line and force it onto the disk immediately."""
    # If a previous run died mid-line, the file does not end with a newline.
    # Start a fresh line first so the broken fragment cannot swallow this record.
    needs_newline = False
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(out_path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# 3. TALKING TO OLLAMA
# ----------------------------------------------------------------------------
def build_messages(post):
    parts = [f"Author: {post.get('author', '')}"]
    if post.get("author_headline"):
        parts.append(f"Author headline: {post['author_headline']}")
    if post.get("attachment_title"):
        parts.append(f"Attached link title: {post['attachment_title']}")
    parts.append("Post text:\n" + post["text_clean"][:MAX_CHARS])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def ask_ollama(model, messages, url):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": ANSWER_SHAPE,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "keep_alive": "30m",
    }
    response = requests.post(f"{url}/api/chat", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def clean_keyphrases(raw_list):
    """Tidy the model's phrases: lowercase, trim, drop empties and duplicates."""
    seen, cleaned = set(), []
    for phrase in raw_list:
        phrase = " ".join(str(phrase).lower().split()).strip(" .,;:-")
        if phrase and phrase not in seen:
            seen.add(phrase)
            cleaned.append(phrase)
    return cleaned


def keyphrases_for(post, model, url, attempts=2):
    """Ask the model; retry once if the answer is unusable. Returns (record, error)."""
    last_error = None
    for _ in range(attempts):
        try:
            data = ask_ollama(model, build_messages(post), url)
            answer = json.loads(data["message"]["content"])
            phrases = clean_keyphrases(answer["keyphrases"])
            if len(phrases) < 3:
                raise ValueError(f"only {len(phrases)} usable keyphrases")
            return {
                "activity_id": post["activity_id"],
                "keyphrases": phrases[:5],
                "model": model,
                "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
            }, None
        except requests.exceptions.ConnectionError:
            sys.exit("Cannot reach Ollama. Is it running? Try: ollama ps")
        except (ValueError, KeyError, requests.exceptions.RequestException) as e:
            # ValueError also covers json.JSONDecodeError
            last_error = f"{type(e).__name__}: {e}"
    return None, last_error


# ----------------------------------------------------------------------------
# 4. THE LOOP
# ----------------------------------------------------------------------------
def fmt_minutes(seconds):
    return f"{int(seconds // 60)} min" if seconds >= 60 else f"{int(seconds)} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N NEW posts this run (for trying it out)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    args = parser.parse_args()

    posts = load_posts(args.csv_path)
    done = load_done_ids(args.out)
    todo = [p for p in posts if p["activity_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Posts with text: {len(posts)} | already done: {len(done)} | "
          f"to do this run: {len(todo)} | model: {args.model}")
    if not todo:
        print("Nothing to do.")
        return

    durations, failures = [], []
    try:
        for i, post in enumerate(todo, start=1):
            started = time.time()
            record, error = keyphrases_for(post, args.model, args.url)
            took = time.time() - started

            if record is None:
                failures.append(post["activity_id"])
                print(f"[{i}/{len(todo)}] FAILED ({error}) - skipped, will retry next run")
                continue

            record["seconds"] = round(took, 1)
            save_result(args.out, record)
            durations.append(took)

            avg = sum(durations) / len(durations)
            eta = avg * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {took:4.1f}s | ETA {fmt_minutes(eta)} | "
                  f"{post.get('author', '')[:22]:22} | {', '.join(record['keyphrases'])}")
    except KeyboardInterrupt:
        print("\nStopped by you. Everything finished so far is saved - just re-run to continue.")

    print(f"\nSaved to {args.out}. Failed this run: {len(failures)}")


if __name__ == "__main__":
    main()