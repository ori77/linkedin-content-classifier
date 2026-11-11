"""Tests for pipeline/step10_gold_sample.py (Step 10): choosing the gold posts and writing the sheet."""
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import classify_inputs as I  # noqa: E402
import rubrics as R  # noqa: E402
import step10_gold_sample as G  # noqa: E402

BASE = 7521000000000000000          # 19-digit ids, like the real ones

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "RAG", "description": "RAG systems.",
         "subtopics": [{"id": "rag.chunking", "name": "Chunking", "description": "Split docs."},
                       {"id": "rag.evaluation", "name": "RAG Evaluation", "description": "Measure RAG."}]},
        {"id": "careers", "name": "Careers", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
        {"id": "agents", "name": "Agents", "description": "Agents.",
         "subtopics": [{"id": "agents.tools", "name": "Tool use", "description": "Tools."}]},
        {"id": "ml", "name": "ML", "description": "ML.",
         "subtopics": [{"id": "ml.basics", "name": "Basics", "description": "Basics."}]},
    ],
    "other": {"id": "other.general", "name": "Other", "description": "Nothing."},
}
SUBS = ["rag.chunking", "rag.evaluation", "careers.interviews", "agents.tools", "ml.basics"]


def words(n, tag="word"):
    return " ".join(f"{tag}{i}" for i in range(n))


def make_world(tmp_path, n=50):
    """50 posts: ids BASE+0..49. 0-4 other (cluster -1); the rest cycle through the 5 subtopics.
    Cluster = 4 for ids divisible by 7, otherwise 100+subtopic number. Posts 10 and 11 state a size."""
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    rows_a, rows_p = [], []
    for i in range(n):
        aid = BASE + i
        if i < 5:
            sub, cluster = "other.general", -1
        else:
            sub = SUBS[i % 5]
            cluster = 4 if i % 7 == 0 else 100 + i % 5
        rows_a.append(f"{aid},{cluster},x,{sub}")
        text = words(30, f"p{i}x")
        if i == 10:
            text += " a 20 questions set"
        if i == 11:
            text += " an 8-week course"
        rows_p.append({"activity_id": aid, "text_clean": text, "attachment_title": f"T{i}",
                       "attachment_site": "site.com", "urls": "https://lnkd.in/x"})
    (tmp_path / "a.csv").write_text("activity_id,cluster,topic_id,subtopic_id\n" + "\n".join(rows_a) + "\n",
                                    encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows_p).to_csv(tmp_path / "p.csv", index=False)
    tax = I.load_taxonomy(tmp_path / "t.yaml")
    assign = I.load_assignments(tmp_path / "a.csv", tax)
    clusters = G.read_clusters(tmp_path / "a.csv")
    posts, _ = I.load_posts(tmp_path / "p.csv")
    return tax, assign, clusters, posts


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


@pytest.fixture
def rub():
    return R.load_rubrics()


def pick(world, **kw):
    tax, assign, clusters, posts = world
    kw.setdefault("generic_clusters", (4,))
    return G.pick_gold(posts, assign, clusters, tax, **kw)


# ------------------------------------------------------------------ the size finder

@pytest.mark.parametrize("text", ["20 questions to prepare", "an 8-week course", "5 hours of video",
                                  "takes 90 minutes", "a 12 week plan", "10 Hrs", "45 mins", "1 question"])
def test_stated_size_found(text):
    assert G.has_stated_size(text)


@pytest.mark.parametrize("text", ["no numbers here", "questionable ideas", "3 weekends away",
                                  "week 3 of the plan", "hours and hours", "2024 was a year"])
def test_stated_size_not_found(text):
    assert not G.has_stated_size(text)


# ------------------------------------------------------------------ clusters

def test_read_clusters_with_bom(tmp_path):
    p = tmp_path / "c.csv"
    p.write_bytes(b"\xef\xbb\xbf" + b"activity_id,cluster,topic_id,subtopic_id\r\n7521877444732116992,3,rag,x\r\n")
    assert G.read_clusters(p) == {7521877444732116992: 3}


def test_read_clusters_needs_cluster_column(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("activity_id,subtopic_id\n1,x\n", encoding="utf-8")
    with pytest.raises(G.GoldError, match="cluster"):
        G.read_clusters(p)


# ------------------------------------------------------------------ choosing

def test_default_mix_is_thirty(world):
    rows, summary = pick(world)
    assert len(rows) == 30 == summary["total"]
    assert summary["got"] == {"stated": 2, "other": 3, "generic": 5, "random": 20}
    assert summary["requested"] == {"stated": 2, "other": 3, "generic": 5, "random": 20}


def test_no_post_twice(world):
    rows, _ = pick(world)
    ids = [r["post"].activity_id for r in rows]
    assert len(ids) == len(set(ids))


def test_gold_ids_in_order(world):
    rows, _ = pick(world)
    assert [r["gold_id"] for r in rows] == [f"g{n:02d}" for n in range(1, 31)]


def test_same_seed_same_posts(world):
    a, _ = pick(world, seed=7)
    b, _ = pick(world, seed=7)
    assert [(r["gold_id"], r["post"].activity_id, r["stratum"]) for r in a] == \
           [(r["gold_id"], r["post"].activity_id, r["stratum"]) for r in b]


def test_other_seed_other_posts(world):
    a, _ = pick(world, seed=1)
    b, _ = pick(world, seed=2)
    assert [r["post"].activity_id for r in a] != [r["post"].activity_id for r in b]


def test_sheet_order_hides_the_groups(world):
    rows, _ = pick(world)
    strata = [r["stratum"] for r in rows]
    changes = sum(1 for a, b in zip(strata, strata[1:]) if a != b)
    assert changes > 3                                    # four groups side by side would change exactly 3 times


def test_stated_group_has_stated_sizes(world):
    rows, _ = pick(world)
    got = [r for r in rows if r["stratum"] == "stated"]
    assert len(got) == 2 and all(G.has_stated_size(r["post"].text) for r in got)
    assert {r["post"].activity_id for r in got} == {BASE + 10, BASE + 11}


def test_other_group_is_other(world):
    rows, _ = pick(world)
    got = [r for r in rows if r["stratum"] == "other"]
    assert len(got) == 3 and all(r["clustering_subtopic"] == I.OTHER_ID for r in got)


def test_generic_group_comes_from_generic_clusters(world):
    rows, _ = pick(world)
    got = [r for r in rows if r["stratum"] == "generic"]
    assert len(got) == 5 and all(r["cluster"] == 4 for r in got)
    assert all(r["clustering_subtopic"] != I.OTHER_ID for r in got)


def test_generic_cluster_list_is_used(world):
    rows, _ = pick(world, generic_clusters=(101,))
    assert all(r["cluster"] == 101 for r in rows if r["stratum"] == "generic")


def test_a_post_matching_two_groups_is_taken_once(world):
    tax, assign, clusters, posts = world
    sized = [I.Post(activity_id=p.activity_id, text=p.text + " 5 hours") for p in posts]
    rows, summary = G.pick_gold(sized, assign, clusters, tax, n_stated=10, n_other=0, n_generic=5, n_random=0,
                                generic_clusters=(4,))
    ids = [r["post"].activity_id for r in rows]
    assert len(ids) == len(set(ids)) == 15 == summary["total"]       # the generic group could not reuse a stated post


def test_generic_group_skips_other_posts(world):
    rows, summary = pick(world, generic_clusters=(-1,), n_other=0)     # -1 is the cluster of the Other posts
    assert summary["got"]["generic"] == 0
    assert all(r["clustering_subtopic"] != I.OTHER_ID for r in rows)


def test_random_group_never_other_and_never_stated_twice(world):
    rows, _ = pick(world)
    rnd = [r for r in rows if r["stratum"] == "random"]
    assert len(rnd) == 20
    assert all(r["clustering_subtopic"] != I.OTHER_ID for r in rnd)


def test_random_group_visits_every_topic_in_turn(world):
    rows, _ = pick(world, n_stated=0, n_other=0, n_generic=0, n_random=4)
    topics = {world[0].topic_of(r["clustering_subtopic"]) for r in rows}
    assert topics == {"rag", "careers", "agents", "ml"}     # 4 posts, 4 topics: one each


def test_short_posts_are_left_out(world):
    tax, assign, clusters, posts = world
    short = [I.Post(activity_id=p.activity_id, text="too short") if p.activity_id == BASE + 10 else p for p in posts]
    rows, _ = G.pick_gold(short, assign, clusters, tax, generic_clusters=(4,))
    assert BASE + 10 not in {r["post"].activity_id for r in rows}


def test_min_words_boundary(world):
    tax, assign, clusters, posts = world
    exact = [I.Post(activity_id=p.activity_id, text=words(20)) if p.activity_id == BASE + 12 else p for p in posts]
    rows, _ = G.pick_gold(exact, assign, clusters, tax, n_stated=0, n_other=0, n_generic=0, n_random=45,
                          generic_clusters=(), min_words=20)
    assert BASE + 12 in {r["post"].activity_id for r in rows}          # 20 words is allowed
    rows, _ = G.pick_gold(exact, assign, clusters, tax, n_stated=0, n_other=0, n_generic=0, n_random=45,
                          generic_clusters=(), min_words=21)
    assert BASE + 12 not in {r["post"].activity_id for r in rows}


def test_posts_without_an_assignment_are_left_out(world):
    tax, assign, clusters, posts = world
    del assign[BASE + 10]
    rows, _ = G.pick_gold(posts, assign, clusters, tax, generic_clusters=(4,))
    assert BASE + 10 not in {r["post"].activity_id for r in rows}


def test_shortfall_goes_to_random(world):
    rows, summary = pick(world, n_stated=5)                 # only 2 posts state a size
    assert summary["got"]["stated"] == 2
    assert summary["got"]["random"] == 23 and len(rows) == 33      # asked 5+3+5+20 = 33, still 33


def test_not_enough_posts_gives_what_exists(world):
    rows, summary = pick(world, n_random=500)
    assert summary["total"] == len(rows) < 500
    assert summary["eligible_posts"] == 50


def test_each_row_carries_the_hidden_facts(world):
    rows, _ = pick(world)
    r = rows[0]
    assert set(r) == {"gold_id", "post", "stratum", "cluster", "clustering_subtopic"}
    assert r["clustering_subtopic"] == world[1][r["post"].activity_id]


def test_topics_in_random_is_counted(world):
    _, summary = pick(world)
    assert summary["topics_in_random"] == 4


# ------------------------------------------------------------------ the sheet

def open_sheet(path):
    from openpyxl import load_workbook
    return load_workbook(path)


@pytest.fixture
def sheet(world, rub, tmp_path):
    rows, _ = pick(world)
    p = tmp_path / "s.xlsx"
    G.write_sheet(p, rows, world[0], rub)
    return open_sheet(p), rows


def test_sheet_has_three_sheets(sheet):
    wb, _ = sheet
    assert wb.sheetnames == ["Label", "Lists", "Guide"]


def test_sheet_headers_and_rows(sheet):
    wb, rows = sheet
    ws = wb["Label"]
    assert [c.value for c in ws[1]] == G.SHEET_COLUMNS
    assert ws.max_row == 31
    assert [ws.cell(row=r, column=1).value for r in range(2, 32)] == [r["gold_id"] for r in rows]


def test_sheet_text_and_attachment(sheet):
    wb, rows = sheet
    ws = wb["Label"]
    p = rows[0]["post"]
    assert ws["B2"].value == p.text
    assert ws["C2"].value == f"{p.attachment_title} | {p.attachment_site}"
    assert ws["D2"].value == p.links


def test_sheet_activity_id_is_exact_text(sheet):
    wb, rows = sheet
    ws = wb["Label"]
    idx = G.SHEET_COLUMNS.index("activity_id") + 1
    for n, r in enumerate(rows, start=2):
        cell = ws.cell(row=n, column=idx)
        assert cell.value == str(r["post"].activity_id) and len(cell.value) == 19
        assert cell.number_format == "@"


def test_sheet_label_columns_start_empty(sheet):
    wb, _ = sheet
    ws = wb["Label"]
    for name in G.LABEL_COLUMNS:
        idx = G.SHEET_COLUMNS.index(name) + 1
        assert all(ws.cell(row=r, column=idx).value is None for r in range(2, 32))


def test_sheet_hides_group_and_clustering_hint(sheet):
    wb, rows = sheet
    ws = wb["Label"]
    headers = [c.value for c in ws[1]]
    assert not any(h in headers for h in ("stratum", "cluster", "clustering_subtopic"))
    body = {str(c.value) for row in ws.iter_rows(min_row=2) for c in row if c.value is not None}
    assert not ({"stated", "other", "generic", "random"} & body)


def test_sheet_dropdowns(sheet):
    wb, _ = sheet
    ws = wb["Label"]
    dvs = {tuple(str(r) for r in dv.sqref.ranges): dv for dv in ws.data_validations.dataValidation}
    by_type = {}
    for dv in ws.data_validations.dataValidation:
        by_type.setdefault(dv.type, []).append(dv)
    assert len(by_type["list"]) == 3 and len(by_type["decimal"]) == 1 and len(by_type["whole"]) == 1
    formulas = sorted(dv.formula1 for dv in by_type["list"])
    assert formulas == ["=Lists!$A$2:$A$7", "=Lists!$B$2:$B$7", "=Lists!$C$2:$C$5"]
    assert any("E2:E31" in str(dv.sqref) for dv in by_type["list"])
    assert "I2:J31" in str(by_type["whole"][0].sqref).replace(" ", "") or \
           {"I2:I31", "J2:J31"} <= set(str(by_type["whole"][0].sqref).split())
    dec = by_type["decimal"][0]
    assert (dec.formula1, dec.formula2) == ("0", "10")
    assert len(dvs) == 5
    assert by_type["whole"][0].formula1 == "1"                    # zero minutes is not allowed


def test_lists_sheet_contents(sheet, world):
    wb, _ = sheet
    lists = wb["Lists"]
    sub = [lists.cell(row=r, column=1).value for r in range(2, 8)]
    assert sub[0] == "rag.chunking | RAG > Chunking" and sub[-1].startswith("other.general | ")
    assert len(sub) == 6 and lists.cell(row=8, column=1).value is None
    assert [lists.cell(row=r, column=2).value for r in range(2, 8)] == R.CONTENT_TYPE_IDS
    assert [lists.cell(row=r, column=3).value for r in range(2, 6)] == R.DIFFICULTY_IDS


def test_subtopic_choices_lead_with_the_id(world):
    choices = G.subtopic_choices(world[0])
    assert [c.split(" | ")[0] for c in choices] == world[0].choice_ids


def test_guide_has_the_rules(sheet, rub):
    wb, _ = sheet
    text = "\n".join(str(c.value) for row in wb["Guide"].iter_rows() for c in row if c.value)
    assert "Judge the resource the post advertises" in text
    assert "AI Engineer" in text
    for d in R.DIFFICULTY_IDS + R.CONTENT_TYPE_IDS:
        assert d in text
    assert "Must-read" in text and "Can an AI Engineer directly apply this" in text
    assert "Bands: 9.0+" in text
    assert "interview preparation" in text and "announces or reports a release" in text.lower()


def test_long_text_is_cut_for_excel(world, rub, tmp_path):
    tax, assign, clusters, posts = world
    big = [I.Post(activity_id=posts[0].activity_id, text="w " * 20000)]
    rows = [{"gold_id": "g01", "post": big[0], "stratum": "random", "cluster": 1, "clustering_subtopic": "rag.chunking"}]
    G.write_sheet(tmp_path / "s.xlsx", rows, tax, rub)
    assert len(open_sheet(tmp_path / "s.xlsx")["Label"]["B2"].value) == G.TEXT_LIMIT


# ------------------------------------------------------------------ files

def test_write_outputs_makes_three_files(world, rub, tmp_path):
    tax = world[0]
    rows, summary = pick(world)
    out = tmp_path / "gold"
    paths = G.write_outputs(out, rows, summary, tax, rub, params={"seed": 42})
    assert all(p.exists() for p in paths.values())
    with paths["key"].open(encoding="utf-8-sig", newline="") as fh:
        key = list(csv.DictReader(fh))
    assert [k["gold_id"] for k in key] == [r["gold_id"] for r in rows]
    assert key[0]["activity_id"] == str(rows[0]["post"].activity_id)
    assert key[0]["stratum"] == rows[0]["stratum"] and key[0]["cluster"] == str(rows[0]["cluster"])
    assert key[0]["clustering_subtopic"] == rows[0]["clustering_subtopic"]
    rec = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert rec["params"] == {"seed": 42} and rec["summary"]["total"] == 30
    assert rec["taxonomy_sha"] == tax.sha[:12] and rec["rubrics_sha"] == rub.sha[:12]
    assert rec["created_utc"].endswith("Z")


def test_never_overwrites_labels(world, rub, tmp_path):
    tax = world[0]
    rows, summary = pick(world)
    G.write_outputs(tmp_path, rows, summary, tax, rub, params={})
    marker = (tmp_path / "gold_labelling.xlsx").read_bytes()
    with pytest.raises(G.GoldError, match="already exist"):
        G.write_outputs(tmp_path, rows, summary, tax, rub, params={})
    assert (tmp_path / "gold_labelling.xlsx").read_bytes() == marker


def test_one_existing_file_is_enough_to_refuse(world, rub, tmp_path):
    rows, summary = pick(world)
    (tmp_path / "gold_key.csv").write_text("x", encoding="utf-8")
    with pytest.raises(G.GoldError, match="gold_key.csv"):
        G.write_outputs(tmp_path, rows, summary, world[0], rub, params={})
    assert not (tmp_path / "gold_labelling.xlsx").exists()                  # nothing half-written


def test_force_overwrites(world, rub, tmp_path):
    rows, summary = pick(world)
    G.write_outputs(tmp_path, rows, summary, world[0], rub, params={})
    G.write_outputs(tmp_path, rows, summary, world[0], rub, params={"again": 1}, force=True)
    assert json.loads((tmp_path / "gold_sample_record.json").read_text(encoding="utf-8"))["params"] == {"again": 1}


# ------------------------------------------------------------------ command line

def run(tmp_path, *extra):
    make_world(tmp_path)
    return G.main(["--posts", str(tmp_path / "p.csv"), "--taxonomy", str(tmp_path / "t.yaml"),
                   "--assignments", str(tmp_path / "a.csv"), "--out-dir", str(tmp_path / "gold"), *extra])


def test_main_end_to_end(tmp_path, capsys):
    assert run(tmp_path, "--generic-clusters", "4") == 0
    out = capsys.readouterr().out
    assert "picked 30 posts" in out and "do NOT open" in out
    assert (tmp_path / "gold" / "gold_labelling.xlsx").exists()
    rec = json.loads((tmp_path / "gold" / "gold_sample_record.json").read_text(encoding="utf-8"))
    assert rec["params"] == {"seed": 42, "min_words": 20, "generic_clusters": [4]}


def key_rows(tmp_path):
    with (tmp_path / "gold" / "gold_key.csv").open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_main_generic_clusters_are_used(tmp_path):
    assert run(tmp_path, "--generic-clusters", "101") == 0
    generic = [k for k in key_rows(tmp_path) if k["stratum"] == "generic"]
    assert len(generic) == 5 and all(k["cluster"] == "101" for k in generic)


def test_main_seed_changes_the_sample(tmp_path):
    run(tmp_path, "--seed", "1", "--generic-clusters", "4")
    first = [k["activity_id"] for k in key_rows(tmp_path)]
    run_dir = tmp_path / "second"
    run_dir.mkdir()
    run(run_dir, "--seed", "2", "--generic-clusters", "4")
    assert [k["activity_id"] for k in key_rows(run_dir)] != first


def test_main_force_overwrites(tmp_path):
    assert run(tmp_path, "--generic-clusters", "4") == 0
    assert G.main(["--posts", str(tmp_path / "p.csv"), "--taxonomy", str(tmp_path / "t.yaml"),
                   "--assignments", str(tmp_path / "a.csv"), "--out-dir", str(tmp_path / "gold"),
                   "--generic-clusters", "4", "--force"]) == 0


def test_main_second_run_refuses(tmp_path, capsys):
    assert run(tmp_path, "--generic-clusters", "4") == 0
    assert G.main(["--posts", str(tmp_path / "p.csv"), "--taxonomy", str(tmp_path / "t.yaml"),
                   "--assignments", str(tmp_path / "a.csv"), "--out-dir", str(tmp_path / "gold")]) == 2
    assert "already exist" in capsys.readouterr().err


def test_main_options_are_used(tmp_path, capsys):
    assert run(tmp_path, "--generic-clusters", "4", "--n-random", "5", "--n-stated", "1", "--n-other", "0",
               "--n-generic", "0", "--seed", "3") == 0
    rec = json.loads((tmp_path / "gold" / "gold_sample_record.json").read_text(encoding="utf-8"))
    assert rec["summary"]["got"] == {"stated": 1, "other": 0, "generic": 0, "random": 5}
    assert rec["params"]["seed"] == 3


def test_main_missing_file_is_exit_2(tmp_path, capsys):
    assert G.main(["--posts", str(tmp_path / "nope.csv"), "--taxonomy", str(tmp_path / "t.yaml"),
                   "--assignments", str(tmp_path / "a.csv"), "--out-dir", str(tmp_path / "g")]) == 2
    assert "cannot make the gold set" in capsys.readouterr().err


def test_main_bad_cluster_list_is_exit_2(tmp_path, capsys):
    assert run(tmp_path, "--generic-clusters", "4,x") == 2
    assert "cannot make the gold set" in capsys.readouterr().err


def test_main_empty_cluster_list_is_fine(tmp_path):
    assert run(tmp_path, "--generic-clusters", "") == 0
