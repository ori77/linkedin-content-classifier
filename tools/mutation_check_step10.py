"""Mutation check for pipeline/step10_gold_sample.py (Step 10).
Run:  python tools/mutation_check_step10.py        (exit 0 = all killed)
"""
import sys

from mutation_runner import run_mutations

MUTATIONS = [
    ("size finder: weeks missing", "weeks?|hours?|hrs?|minutes?|mins?|questions?", "hours?|hrs?|minutes?|mins?|questions?"),
    ("size finder: questions missing", "|mins?|questions?)", "|mins?)"),
    ("size finder: word boundary dropped", "|questions?)\\b\"", "|questions?)\""),
    ("size finder: case matters", "questions?)\\b\", re.I)", "questions?)\\b\")"),
    ("clusters: BOM not skipped", 'Path(path).open(newline="", encoding="utf-8-sig")', 'Path(path).open(newline="", encoding="utf-8")'),
    ("clusters: missing column accepted", 'if "cluster" not in (reader.fieldnames or []):', "if False:"),
    ("pick: seed ignored", "rng = random.Random(seed)", "rng = random.Random()"),
    ("pick: short posts allowed", "p.word_count >= min_words", "p.word_count >= 0"),
    ("pick: min words boundary", "p.word_count >= min_words", "p.word_count > min_words"),
    ("pick: unassigned posts allowed", "if p.activity_id in assign and", "if True and"),
    ("pick: duplicates allowed", "if p.activity_id not in chosen and matches(p)]", "if matches(p)]"),
    ("pick: stated group ignores sizes", "draw(\"stated\", n_stated, lambda p: has_stated_size(p.text))",
     "draw(\"stated\", n_stated, lambda p: True)"),
    ("pick: other group takes any post", "lambda p: assign[p.activity_id] == I.OTHER_ID)", "lambda p: True)"),
    ("pick: generic group ignores clusters", "clusters.get(p.activity_id) in generic_clusters and assign[p.activity_id] != I.OTHER_ID",
     "assign[p.activity_id] != I.OTHER_ID"),
    ("pick: generic group may take other posts", "in generic_clusters and assign[p.activity_id] != I.OTHER_ID", "in generic_clusters"),
    ("pick: shortfall lost", "wanted_random = n_random + shortfall", "wanted_random = n_random"),
    ("pick: random group may take other posts", "if p.activity_id not in chosen and topic != I.OTHER_TOPIC_ID:",
     "if p.activity_id not in chosen:"),
    ("pick: random takes already chosen posts", "if p.activity_id not in chosen and topic != I.OTHER_TOPIC_ID:",
     "if topic != I.OTHER_TOPIC_ID:"),
    ("pick: topics not taken in turn", "chosen[by_topic[t].pop().activity_id] = \"random\"\n                taken += 1",
     "chosen[by_topic[t].pop().activity_id] = \"random\"\n                taken += 1\n                break"),
    ("pick: sheet order not shuffled", "rng.shuffle(order)                                 # the sheet order hides the group", "pass"),
    ("pick: gold ids start at 0", "enumerate(order, start=1)", "enumerate(order, start=0)"),
    ("pick: stratum not recorded", '"stratum": chosen[aid],', '"stratum": "random",'),
    ("pick: cluster not recorded", '"cluster": clusters.get(aid, -1)', '"cluster": -1'),
    ("pick: topics_in_random wrong", 'for r in rows if r["stratum"] == "random"})}', 'for r in rows})}'),
    ("sheet: attachment site dropped", "for x in (p.attachment_title, p.attachment_site) if x", "for x in (p.attachment_title,) if x"),
    ("sheet: other.general not offered", '+ [f"{I.OTHER_ID} | Other / does not fit any subtopic"]', "+ []"),
    ("sheet: activity id not text", 'number_format = "@"', 'number_format = "General"'),
    ("sheet: activity id as number", "str(p.activity_id)])", "p.activity_id])"),
    ("sheet: text not cut", "p.text[:TEXT_LIMIT]", "p.text"),
    ("sheet: subtopic dropdown missing", 'add_list("my_subtopic", f"=Lists!$A$2:$A${len(choices) + 1}")', "pass"),
    ("sheet: type dropdown wrong range", "=Lists!$B$2:$B${len(types) + 1}", "=Lists!$B$2:$B${len(types)}"),
    ("sheet: difficulty dropdown wrong column", "=Lists!$C$2:$C${len(levels) + 1}", "=Lists!$B$2:$B${len(levels) + 1}"),
    ("sheet: alignment range wrong", 'formula1="0", formula2="10"', 'formula1="0", formula2="100"'),
    ("sheet: minutes may be zero", 'formula1="1", allow_blank=True', 'formula1="0", allow_blank=True'),
    ("sheet: understand minutes unchecked", 'for name in ("my_cover_minutes", "my_understand_minutes"):', 'for name in ("my_cover_minutes",):'),
    ("sheet: last row off by one", "last = len(rows) + 1", "last = len(rows)"),
    ("guide: judging basis missing", 'd["judging_basis"].strip(), "",', '"",'),
    ("guide: bands missing", '"  Bands: " + "; ".join(', '"" + "; ".join('),
    ("guide: content types missing", "' '.join(x['rule'].split())", "''"),
    ("files: overwrite allowed", "if existing and not force:", "if False:"),
    ("files: force ignored", "if existing and not force:", "if existing:"),
    ("files: key without strata", 'r["stratum"], r["cluster"], r["clustering_subtopic"]])', 'r["cluster"], r["clustering_subtopic"]])'),
    ("files: record without taxonomy hash", '"taxonomy_sha": tax.sha[:12],', '"taxonomy_sha": "",'),
    ("main: errors give exit 0", "        return 2\n    print(", "        return 0\n    print("),
    ("main: generic clusters ignored", "generic_clusters=generic, min_words=args.min_words)", "generic_clusters=DEFAULT_GENERIC_CLUSTERS, min_words=args.min_words)"),
    ("main: seed ignored", "pick_gold(posts, assign, clusters, tax, seed=args.seed,", "pick_gold(posts, assign, clusters, tax, seed=42,"),
    ("main: force ignored", "params=params, force=args.force)", "params=params)"),
]

if __name__ == "__main__":
    sys.exit(run_mutations("pipeline/step10_gold_sample.py", "tests/test_step10_gold_sample.py", MUTATIONS))
