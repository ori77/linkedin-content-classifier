"""Step 10: pick the gold-set posts and write the labelling sheet.

In simple terms
---------------
The "gold set" is a small group of posts (about 30) that YOU label by hand. Later the
classifier's answers are compared with yours, so we can see where it is wrong and tune the
confidence rules. This script does the picking and prepares an Excel sheet to fill in.

How the 30 posts are chosen (every number can be changed on the command line):
  * 2  posts that state a size ("20 questions", "8-week course")   -> checks the time conversion
  * 3  posts clustering put in "Other"                              -> checks the other.general policy
  * 5  posts from clusters flagged as generic                       -> where misfiling is most likely
  * 20 random posts, one topic after another in turn                -> a fair spot-check; rare topics get a turn too
A stratum with fewer candidates than asked gives what it has, and the shortfall goes to the
random group, so the total stays what you asked for.

Same seed + same inputs = same posts. The order in the sheet is shuffled, so the sheet does not
reveal which group a post came from.

Files written to --out-dir (and never overwritten without --force, because your labels live there):
  gold_labelling.xlsx    the sheet you fill in (dropdowns for subtopic, type, difficulty)
  gold_key.csv           which post is which, its group and what clustering said. DO NOT OPEN before you
                         have labelled: it would bias your labels.
  gold_sample_record.json how the sample was made (seed, settings, counts, input fingerprints)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import classify_inputs as I
import rubrics as R

DEFAULT_GENERIC_CLUSTERS = (4, 5, 10, 15, 25, 27, 31, 32, 36)   # flagged "looks generic" in step 5 v2.1
DEFAULT_MIN_WORDS = 20            # shorter posts are too thin to label
SHEET_NAME = "Label"
TEXT_LIMIT = 30000                # Excel allows 32,767 characters in one cell; posts are capped at 12,000 anyway

# "20 questions", "8-week", "5 hours", "90 minutes" ... a rough finder, not the classifier's job
STATED_SIZE_RE = re.compile(r"\b\d+\s*[-\s]?\s*(?:weeks?|hours?|hrs?|minutes?|mins?|questions?)\b", re.I)

LABEL_COLUMNS = ["my_subtopic", "my_content_type", "my_difficulty", "my_alignment",
                 "my_cover_minutes", "my_understand_minutes", "notes"]
SHEET_COLUMNS = ["gold_id", "post", "attachment", "links"] + LABEL_COLUMNS + ["activity_id"]


class GoldError(Exception):
    """The sample cannot be made; the message says why."""


# ------------------------------------------------------------------ choosing the posts

def has_stated_size(text: str) -> bool:
    return STATED_SIZE_RE.search(text) is not None


def read_clusters(path: Path | str) -> dict[int, int]:
    """activity_id -> cluster number (-1 = noise), from taxonomy_assignments.v1.csv."""
    result: dict[int, int] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "cluster" not in (reader.fieldnames or []):
            raise GoldError("the assignments file has no 'cluster' column")
        for row in reader:
            result[int(row["activity_id"])] = int(row["cluster"])
    return result


def pick_gold(posts: list[I.Post], assign: dict[int, str], clusters: dict[int, int], tax: I.Taxonomy, *,
              seed: int = 42, n_stated: int = 2, n_other: int = 3, n_generic: int = 5, n_random: int = 20,
              generic_clusters: tuple[int, ...] = DEFAULT_GENERIC_CLUSTERS,
              min_words: int = DEFAULT_MIN_WORDS) -> tuple[list[dict], dict]:
    """Return (picked rows in sheet order, summary). Each row: gold_id, post, stratum, cluster, clustering_subtopic."""
    rng = random.Random(seed)
    eligible = sorted((p for p in posts if p.activity_id in assign and p.word_count >= min_words),
                      key=lambda p: p.activity_id)
    chosen: dict[int, str] = {}                       # activity_id -> stratum
    by_id = {p.activity_id: p for p in eligible}

    def draw(stratum: str, wanted: int, matches) -> int:
        pool = [p for p in eligible if p.activity_id not in chosen and matches(p)]
        got = rng.sample(pool, min(wanted, len(pool)))
        for p in got:
            chosen[p.activity_id] = stratum
        return wanted - len(got)                       # shortfall

    shortfall = 0
    shortfall += draw("stated", n_stated, lambda p: has_stated_size(p.text))
    shortfall += draw("other", n_other, lambda p: assign[p.activity_id] == I.OTHER_ID)
    shortfall += draw("generic", n_generic,
                      lambda p: clusters.get(p.activity_id) in generic_clusters and assign[p.activity_id] != I.OTHER_ID)

    # random group: take one post from each topic in turn, so small topics are not drowned out
    wanted_random = n_random + shortfall
    by_topic: dict[str, list[I.Post]] = {}
    for p in eligible:
        topic = tax.topic_of(assign[p.activity_id])
        if p.activity_id not in chosen and topic != I.OTHER_TOPIC_ID:
            by_topic.setdefault(topic, []).append(p)
    topics = sorted(by_topic)
    rng.shuffle(topics)
    for t in topics:
        rng.shuffle(by_topic[t])
    taken = 0
    while taken < wanted_random and any(by_topic.values()):
        for t in topics:
            if taken >= wanted_random:
                break
            if by_topic[t]:
                chosen[by_topic[t].pop().activity_id] = "random"
                taken += 1

    order = list(chosen)
    rng.shuffle(order)                                 # the sheet order hides the group
    rows = []
    for n, aid in enumerate(order, start=1):
        rows.append({"gold_id": f"g{n:02d}", "post": by_id[aid], "stratum": chosen[aid],
                     "cluster": clusters.get(aid, -1), "clustering_subtopic": assign[aid]})
    counts = {s: sum(1 for r in rows if r["stratum"] == s) for s in ("stated", "other", "generic", "random")}
    summary = {"requested": {"stated": n_stated, "other": n_other, "generic": n_generic, "random": n_random},
               "got": counts, "total": len(rows), "eligible_posts": len(eligible),
               "topics_in_random": len({tax.topic_of(r["clustering_subtopic"]) for r in rows if r["stratum"] == "random"})}
    return rows, summary


# ------------------------------------------------------------------ the sheet

def _attachment(p: I.Post) -> str:
    return " | ".join(x for x in (p.attachment_title, p.attachment_site) if x)


def subtopic_choices(tax: I.Taxonomy) -> list[str]:
    """Dropdown entries: 'id | Topic > Name'. The id before ' | ' is what is read back later."""
    return [f"{s.id} | {s.label}" for s in tax.subtopics] + [f"{I.OTHER_ID} | Other / does not fit any subtopic"]


def write_sheet(path: Path, rows: list[dict], tax: I.Taxonomy, rub: R.Rubrics) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    lists = wb.create_sheet("Lists")
    guide = wb.create_sheet("Guide")

    choices = subtopic_choices(tax)
    types = list(R.CONTENT_TYPE_IDS)
    levels = list(R.DIFFICULTY_IDS)
    for col, (title, items) in enumerate((("subtopic", choices), ("content_type", types), ("difficulty", levels)), 1):
        lists.cell(row=1, column=col, value=title).font = Font(bold=True)
        for r, item in enumerate(items, start=2):
            lists.cell(row=r, column=col, value=item)
    lists.column_dimensions["A"].width = 70

    ws.append(SHEET_COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    for row in rows:
        p: I.Post = row["post"]
        ws.append([row["gold_id"], p.text[:TEXT_LIMIT], _attachment(p), p.links,
                   None, None, None, None, None, None, None, str(p.activity_id)])
        ws.cell(row=ws.max_row, column=SHEET_COLUMNS.index("activity_id") + 1).number_format = "@"   # keep 19 digits as text
    last = len(rows) + 1

    def col(name: str) -> str:
        from openpyxl.utils import get_column_letter
        return get_column_letter(SHEET_COLUMNS.index(name) + 1)

    def add_list(name: str, source: str) -> None:
        dv = DataValidation(type="list", formula1=source, allow_blank=True, showErrorMessage=True,
                            errorTitle="Pick from the list", error="Please choose a value from the dropdown.")
        ws.add_data_validation(dv)
        dv.add(f"{col(name)}2:{col(name)}{last}")

    add_list("my_subtopic", f"=Lists!$A$2:$A${len(choices) + 1}")
    add_list("my_content_type", f"=Lists!$B$2:$B${len(types) + 1}")
    add_list("my_difficulty", f"=Lists!$C$2:$C${len(levels) + 1}")
    dv = DataValidation(type="decimal", operator="between", formula1="0", formula2="10", allow_blank=True,
                        showErrorMessage=True, errorTitle="0 to 10", error="Enter a number from 0 to 10.")
    ws.add_data_validation(dv)
    dv.add(f"{col('my_alignment')}2:{col('my_alignment')}{last}")
    dv = DataValidation(type="whole", operator="greaterThanOrEqual", formula1="1", allow_blank=True,
                        showErrorMessage=True, errorTitle="Minutes", error="Enter a whole number of minutes (1 or more).")
    ws.add_data_validation(dv)
    for name in ("my_cover_minutes", "my_understand_minutes"):
        dv.add(f"{col(name)}2:{col(name)}{last}")

    widths = {"gold_id": 8, "post": 90, "attachment": 28, "links": 22, "my_subtopic": 42, "my_content_type": 20,
              "my_difficulty": 14, "my_alignment": 12, "my_cover_minutes": 14, "my_understand_minutes": 16,
              "notes": 30, "activity_id": 22}
    for name, width in widths.items():
        ws.column_dimensions[col(name)].width = width
    for r in range(2, last + 1):
        for name in ("post", "attachment", "links", "notes"):
            ws.cell(row=r, column=SHEET_COLUMNS.index(name) + 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    _write_guide(guide, rub)
    wb.save(path)


def _write_guide(ws, rub: R.Rubrics) -> None:
    d = rub.data
    lines: list[str] = ["HOW TO LABEL (read once)", "",
                        "Fill the green columns of the 'Label' sheet. Use only what the post itself says.", "",
                        "TIME (columns my_cover_minutes and my_understand_minutes)",
                        d["judging_basis"].strip(), "",
                        "  cover = time to get through the resource once; understand = time to really learn it.",
                        "  If you are unsure, give your best guess in minutes; leave blank only if you cannot guess.", "",
                        "ALIGNMENT (my_alignment, 0 to 10): how valuable is this for someone working as: " + d["role_profile"],
                        "  Think about these six questions (the classifier uses the same ones):"]
    for a in d["alignment"]:
        lines.append(f"  - {a['question']}  high = {a['high']}; low = {a['low']}")
    lines += ["  Bands: " + "; ".join(f"{b['min']}+ = {b['label']}" for b in d["alignment_bands"]), "", "DIFFICULTY"]
    for x in d["difficulty"]:
        lines.append(f"  - {x['id']}: {x['definition']}  e.g. {x['example']}")
    lines += ["", "CONTENT TYPE (the FIRST rule that applies wins, top to bottom)"]
    for x in d["content_type"]:
        lines.append(f"  - {x['id']}: {' '.join(x['rule'].split())}")
    for n, text in enumerate(lines, start=1):
        ws.cell(row=n, column=1, value=text)
    ws.column_dimensions["A"].width = 160


# ------------------------------------------------------------------ writing everything

def write_outputs(out_dir: Path, rows: list[dict], summary: dict, tax: I.Taxonomy, rub: R.Rubrics, *,
                  params: dict, force: bool = False) -> dict[str, Path]:
    out_dir = Path(out_dir)
    paths = {"sheet": out_dir / "gold_labelling.xlsx", "key": out_dir / "gold_key.csv",
             "record": out_dir / "gold_sample_record.json"}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not force:
        raise GoldError(f"{', '.join(existing)} already exist in {out_dir}; your labels may be in them. "
                        "Use --force only if you really want to start over.")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sheet(paths["sheet"], rows, tax, rub)
    with paths["key"].open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["gold_id", "activity_id", "stratum", "cluster", "clustering_subtopic"])
        for r in rows:
            w.writerow([r["gold_id"], r["post"].activity_id, r["stratum"], r["cluster"], r["clustering_subtopic"]])
    record = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "params": params,
              "summary": summary, "taxonomy_sha": tax.sha[:12], "rubrics_sha": rub.sha[:12]}
    paths["record"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pick the gold-set posts and write the labelling sheet.")
    ap.add_argument("--posts", required=True, help="posts_with_save_dates.csv")
    ap.add_argument("--taxonomy", required=True, help="taxonomy.v1.yaml")
    ap.add_argument("--assignments", required=True, help="taxonomy_assignments.v1.csv")
    ap.add_argument("--rubrics", default=str(R.DEFAULT_PATH))
    ap.add_argument("--out-dir", default="gold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-stated", type=int, default=2)
    ap.add_argument("--n-other", type=int, default=3)
    ap.add_argument("--n-generic", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--generic-clusters", default=",".join(map(str, DEFAULT_GENERIC_CLUSTERS)),
                    help="comma-separated cluster numbers flagged as generic")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--force", action="store_true", help="overwrite existing gold files (destroys labels!)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generic = tuple(int(x) for x in args.generic_clusters.split(",") if x.strip())
        tax = I.load_taxonomy(args.taxonomy)
        assign = I.load_assignments(args.assignments, tax)
        clusters = read_clusters(args.assignments)
        posts, _skipped = I.load_posts(args.posts)
        rub = R.load_rubrics(args.rubrics)
        rows, summary = pick_gold(posts, assign, clusters, tax, seed=args.seed, n_stated=args.n_stated,
                                  n_other=args.n_other, n_generic=args.n_generic, n_random=args.n_random,
                                  generic_clusters=generic, min_words=args.min_words)
        params = {"seed": args.seed, "min_words": args.min_words, "generic_clusters": list(generic)}
        paths = write_outputs(Path(args.out_dir), rows, summary, tax, rub, params=params, force=args.force)
    except (I.InputError, R.RubricsError, GoldError, ValueError, OSError) as exc:
        print(f"cannot make the gold set: {exc}", file=sys.stderr)
        return 2
    print(f"picked {summary['total']} posts: {summary['got']} (asked {summary['requested']})")
    print(f"sheet to fill in : {paths['sheet']}")
    print(f"key (do NOT open until you have labelled): {paths['key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
