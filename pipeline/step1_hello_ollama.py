"""
Step 1: send ONE real post to a local Ollama model and print what comes back.

Goal of this step: understand the plumbing (Python -> Ollama -> Python) before we
build anything bigger. Nothing is saved anywhere; it only prints.

Run (Windows CMD):
    pip install requests
    python step1_hello_ollama.py "posts_with_save_dates.csv"
    python step1_hello_ollama.py "posts_with_save_dates.csv" --row 5
    python step1_hello_ollama.py "posts_with_save_dates.csv" --model qwen3:4b
"""
import argparse
import csv
import json
import sys
import time

import requests

# Windows CMD often uses an old text encoding (cp1252) that crashes on emojis and
# styled letters that LinkedIn posts are full of. This makes printing safe.
# sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# 1. SETTINGS: things you might want to change live at the top
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434"   # where Ollama listens (only your own laptop)
DEFAULT_MODEL = "qwen3:latest"          # the 8B model you have
NUM_CTX = 4096                          # how much text the model can hold at once (tokens)
MAX_CHARS = 3500                        # LinkedIn posts max out near 3,000 characters

# The shape of the answer we want back. Ollama will force the model to follow it.
ANSWER_SHAPE = {
    "type": "object",
    "properties": {
        "main_topic": {"type": "string"},
        "one_sentence_summary": {"type": "string"},
        "is_tutorial_or_news_or_opinion": {
            "type": "string",
            "enum": ["tutorial", "news", "opinion", "interview_questions", "other"],
        },
    },
    "required": ["main_topic", "one_sentence_summary", "is_tutorial_or_news_or_opinion"],
}


# ----------------------------------------------------------------------------
# 2. READ ONE POST from the CSV we built earlier
# ----------------------------------------------------------------------------
def load_post(csv_path, row_number):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))          # every row becomes a dict: {"author": ..., "text": ...}
    if not 0 <= row_number < len(rows):
        sys.exit(f"Row {row_number} doesn't exist. The file has {len(rows)} rows (0 to {len(rows) - 1}).")
    return rows[row_number]


# ----------------------------------------------------------------------------
# 3. BUILD THE MESSAGES we send to the model
# ----------------------------------------------------------------------------
def build_messages(post):
    system_rules = (
        "You read a LinkedIn post and describe it briefly. "
        "Base your answer only on the post text. Reply only in the requested JSON format."
    )
    text = (post.get("text_clean") or post.get("text") or "")[:MAX_CHARS]
    user_message = (
        f"AUTHOR: {post.get('author', '')}\n"
        f"AUTHOR HEADLINE: {post.get('author_headline', '')}\n"
        f"ATTACHED LINK TITLE: {post.get('attachment_title', '') or '(none)'}\n\n"
        f"POST TEXT:\n{text}"
    )
    return [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": user_message},
    ]


# ----------------------------------------------------------------------------
# 4. TALK TO OLLAMA
# ----------------------------------------------------------------------------
def ask_ollama(model, messages, url=OLLAMA_URL):
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,              # give me the whole answer at once, not word by word
        "think": False,               # Qwen3 can "think out loud" first; off = faster, cleaner
        "format": ANSWER_SHAPE,       # force the answer to match the shape above
        "options": {
            "temperature": 0,         # 0 = no randomness, same post -> same answer
            "num_ctx": NUM_CTX,       # context window size
        },
        "keep_alive": "30m",          # keep the model in memory for 30 min after use
    }
    try:
        response = requests.post(f"{url}/api/chat", json=request_body, timeout=600)
    except requests.exceptions.ConnectionError:
        sys.exit("Can't reach Ollama. Is it running? (Open the Ollama app, or run `ollama serve`.)")
    if response.status_code != 200:
        sys.exit(f"Ollama returned an error ({response.status_code}): {response.text[:500]}")
    return response.json()


# ----------------------------------------------------------------------------
# 5. MAIN: put the pieces together and print everything we learn
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--row", type=int, default=0, help="which post to use (0 = first)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    args = parser.parse_args()

    post = load_post(args.csv_path, args.row)
    words = int(float(post.get("word_count") or 0))   # the CSV may store 141 as "141.0"
    print(f"Post #{args.row} by {post['author']}  ({words} words)")
    print("-" * 60)
    print((post.get("text_clean") or "")[:400].replace("\n", " "), "...\n")

    messages = build_messages(post)
    print(f"Sending to {args.model} ... (the first call also loads the model, so it can be slow)")
    started = time.time()
    reply = ask_ollama(args.model, messages, args.url)
    seconds = time.time() - started

    raw_answer = reply["message"]["content"]          # the model's answer, as text
    print("\nRAW ANSWER (exactly what the model sent back):")
    print(raw_answer)

    answer = json.loads(raw_answer)                    # turn the text into a Python dict
    print("\nPARSED (now it's a normal Python dictionary):")
    for key, value in answer.items():
        print(f"  {key}: {value}")

    tokens_in = reply.get("prompt_eval_count", 0)      # how many tokens the model read
    tokens_out = reply.get("eval_count", 0)            # how many tokens it wrote
    print("\nSTATS:")
    print(f"  total time      : {seconds:.1f} s")
    print(f"  tokens read     : {tokens_in}")
    print(f"  tokens written  : {tokens_out}")
    if reply.get("eval_duration"):
        print(f"  writing speed   : {tokens_out / (reply['eval_duration'] / 1e9):.1f} tokens/second")
    print("\nNow run `ollama ps` in another window to see where the model is sitting.")


if __name__ == "__main__":
    main()