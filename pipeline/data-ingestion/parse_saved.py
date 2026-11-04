"""
parse_saved.py: turn LinkedIn "Saved posts" page captures into one clean CSV.

What it reads
-------------
The saved-posts list lives inside an <iframe> on LinkedIn's page. When you save
the page with Ctrl+S -> "Webpage, Complete", that iframe is written as a separate
file inside the "..._files" folder, e.g. "saved_resource(4).html". It is usually
the LARGEST saved_resource(N).html in that folder.

You can pass either:
  * the frame file(s) directly           -> saved_resource(4).html
  * the main page file(s)                -> Saved Posts _ LinkedIn_01.html
    (the script follows the "interop-iframe" pointer to the frame file)
  * a folder                             -> every .html inside it is scanned

Each post "card" in the frame file is a <div data-chameleon-result-urn=
"urn:li:activity:<ID>">. Everything about that post sits inside that div.

What it writes (one row per unique post)
----------------------------------------
activity_id        LinkedIn's post ID (same number as in the export CSV URL)
post_url           https://www.linkedin.com/feed/update/urn:li:activity:<ID>
post_date_utc      publish date, decoded from the ID (no internet needed)
relative_time      what the card showed, e.g. "1d", "3w" (relative to capture day)
author             post author name
author_headline    author's LinkedIn headline
text               full post text, line breaks kept
text_clean         same text, Unicode-normalised (fancy bold/italic letters -> plain)
word_count, char_count
urls               every URL found in the post text, unique, " | " separated
url_count
attachment_title   title of an attached link preview (article, GitHub repo, ...)
attachment_site    site shown under that title, e.g. "github.com"
has_image          True if the card shows an image/thumbnail
source_file        which capture file the row came from

Nothing here touches the internet or LinkedIn. It only reads files on disk.

Usage
-----
python parse_saved.py OUTPUT.csv INPUT [INPUT ...]
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

CARD_ATTR = "data-chameleon-result-urn"
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
FIELDS = [
    "activity_id", "post_url", "post_date_utc", "relative_time", "author",
    "author_headline", "text", "text_clean", "word_count", "char_count",
    "urls", "url_count", "attachment_title", "attachment_site", "has_image",
    "source_file",
]


# ---------------------------------------------------------------- helpers
def squash(t: str) -> str:
    """Collapse all whitespace to single spaces."""
    return re.sub(r"\s+", " ", t or "").strip()


def activity_date(aid: int) -> str:
    """LinkedIn IDs start with a millisecond timestamp in their top 41 bits."""
    return datetime.fromtimestamp((aid >> 22) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def text_with_breaks(node: Tag) -> str:
    """Get text from a node, turning <br> into new lines and skipping buttons."""
    parts: list[str] = []
    for el in node.descendants:
        if isinstance(el, Tag):
            if el.name == "br":
                parts.append("\n")
        elif isinstance(el, NavigableString):
            if el.find_parent("button") is not None:      # "…see more" button
                continue
            if el.__class__.__name__ == "Comment":         # <!-- --> markers
                continue
            parts.append(str(el))
    raw = "".join(parts)
    lines = [squash(line) for line in raw.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return re.sub(r"\s*…\s*see more\s*$", "", text, flags=re.I).strip()


def frame_files(path: Path) -> list[Path]:
    """Resolve an input path to the HTML file(s) that actually hold post cards."""
    if path.is_dir():
        return sorted(p for p in path.rglob("*.html"))
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if CARD_ATTR in raw:
        return [path]
    # main page: follow the interop-iframe pointer into the _files folder
    m = re.search(r'data-testid="interop-iframe"[^>]*src="([^"]+)"', raw) or \
        re.search(r'<iframe[^>]*src="([^"]+)"[^>]*data-testid="interop-iframe"', raw)
    if m:
        target = (path.parent / m.group(1)).resolve()
        if target.exists():
            return [target]
        print(f"  ! frame file not found: {target}", file=sys.stderr)
    return []


# ---------------------------------------------------------------- parsing
def parse_card(card: Tag, source: str) -> dict:
    aid = int(card[CARD_ATTR].rsplit(":", 1)[1])

    actor = card.select_one(".entity-result__content-actor")
    name = actor.select_one("a span[dir=ltr]") if actor else None
    headline = actor.select_one(".linked-area > div") if actor else None
    rel = actor.select_one(".linked-area p") if actor else None
    rel_txt = squash(rel.get_text(" ")).rstrip("• ").strip() if rel else ""

    # Post text. Normal cards: .entity-result__content-summary.
    # Cards with an attachment: the text is the "linked-area" block inside the
    # inner container; the attachment is a sibling block we read separately.
    body = card.select_one(".entity-result__content-summary")
    if body is None:
        inner = card.select_one(".entity-result__content-inner-container--right-padding")
        body = inner.select_one(":scope > .linked-area") if inner else None
    text = text_with_breaks(body) if body else ""

    att_title = att_site = ""
    att = card.select_one(".entity-result__embedded-object")
    if att:
        t = att.select_one(".entity-result__embedded-object-title")
        s = att.select_one(".entity-result__embedded-object-sub-title")
        att_title = squash(t.get_text(" ")) if t else ""
        att_site = squash(s.get_text(" ")) if s else ""

    clean = unicodedata.normalize("NFKC", text)
    urls = list(dict.fromkeys(u.rstrip(".,;:!?") for u in URL_RE.findall(clean)))

    return {
        "activity_id": aid,
        "post_url": f"https://www.linkedin.com/feed/update/urn:li:activity:{aid}",
        "post_date_utc": activity_date(aid),
        "relative_time": rel_txt,
        "author": squash(name.get_text()) if name else "",
        "author_headline": squash(headline.get_text(" ")) if headline else "",
        "text": text,
        "text_clean": clean,
        "word_count": len(clean.split()),
        "char_count": len(clean),
        "urls": " | ".join(urls),
        "url_count": len(urls),
        "attachment_title": att_title,
        "attachment_site": att_site,
        "has_image": bool(card.select_one("img.entity-result__embedded-object-image, "
                                          ".entity-result__embedded-object-image")),
        "source_file": source,
    }


def parse_file(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    return [parse_card(c, path.name) for c in soup.find_all(attrs={CARD_ATTR: True})]


def main(out_csv: str, inputs: list[str]) -> None:
    best: dict[int, dict] = {}
    for inp in inputs:
        for f in frame_files(Path(inp)):
            rows = parse_file(f)
            if not rows:
                continue
            print(f"  {f.name}: {len(rows)} cards")
            for r in rows:   # de-duplicate: keep the copy with the most text
                old = best.get(r["activity_id"])
                if old is None or r["char_count"] > old["char_count"]:
                    best[r["activity_id"]] = r
    rows = sorted(best.values(), key=lambda r: r["activity_id"], reverse=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} unique posts -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
