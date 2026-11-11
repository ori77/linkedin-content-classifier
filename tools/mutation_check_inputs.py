"""Mutation check for pipeline/classify_inputs.py (Step 9).
Run:  python tools/mutation_check_inputs.py        (exit 0 = all killed)
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    ("file_sha: line endings matter", 'raw.replace(b"\\r\\n", b"\\n").replace(b"\\r", b"\\n")', "raw"),
    ("text check: blanks count as text", 'isinstance(x, str) and x.strip() != ""', 'isinstance(x, str) and x != ""'),
    ("label: wrong separator", 'f"{self.topic_name} > {self.name}"', 'f"{self.topic_name} - {self.name}"'),
    ("choices: other.general missing", "+ [OTHER_ID]", "+ []"),
    ("topic_of: other.general not special", "if subtopic_id == OTHER_ID:", "if False:"),
    ("taxonomy: version unchecked", 'if not _text(data.get("version")):', "if False:"),
    ("taxonomy: empty topic list accepted", "if not isinstance(topics, list) or not topics:", "if not isinstance(topics, list):"),
    ("taxonomy: duplicate topic id accepted", "if tid in seen_topics:", "if False:"),
    ("taxonomy: reserved topic id accepted", "if tid == OTHER_TOPIC_ID:", "if False:"),
    ("taxonomy: topic without subtopics accepted", "if not isinstance(subs, list) or not subs:", "if not isinstance(subs, list):"),
    ("taxonomy: duplicate subtopic id accepted", "if sid in seen_sub:", "if False:"),
    ("taxonomy: id/prefix mismatch accepted", 'if not sid.startswith(tid + "."):', "if False:"),
    ("taxonomy: bad keywords accepted", "if not isinstance(kw, list) or any(not _text(k) for k in kw):", "if False:"),
    ("taxonomy: 'other' entry optional", 'if not isinstance(other, dict) or other.get("id") != OTHER_ID:', "if False:"),
    ("taxonomy: descriptions keep line breaks", 'description=" ".join(sub["description"].split())', 'description=sub["description"]'),
    ("assignments: BOM not skipped", 'path.open(newline="", encoding="utf-8-sig")', 'path.open(newline="", encoding="utf-8")'),
    ("assignments: missing columns accepted", "if missing:\n                raise InputError", "if False:\n                raise InputError"),
    ("assignments: unknown subtopic accepted", "if sid not in valid:", "if False:"),
    ("assignments: duplicate id accepted", "if aid in result:", "if False:"),
    ("posts: raw text never used", 'or _clean(row.get("text"))', ""),
    ("posts: 'nan' counts as text", '("nan", "none", "<na>")', '("none", "<na>")'),
    ("posts: duplicate id accepted", "if aid in seen:", "if False:"),
    ("posts: long text not cut", "text=text[:MAX_POST_CHARS]", "text=text"),
    ("posts: limit boundary", "truncated = len(text) > MAX_POST_CHARS", "truncated = len(text) >= MAX_POST_CHARS"),
    ("posts: links not cut", '_clean(row.get("urls"))[:300]', '_clean(row.get("urls"))'),
    ("posts: rows without text classified", 'skipped.append((aid, "no text"))\n            continue', 'skipped.append((aid, "no text"))'),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/classify_inputs.py", "tests/test_classify_inputs.py", MUTATIONS))
