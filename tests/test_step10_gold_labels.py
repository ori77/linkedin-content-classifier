"""Tests for pipeline/step10_gold_labels.py (Step 10b): reading and checking the labelled gold sheet."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_inputs as I  # noqa: E402
import rubrics as R  # noqa: E402
import step10_gold_labels as L  # noqa: E402
import step10_gold_sample as G  # noqa: E402
from test_step10_gold_sample import BASE, make_world  # noqa: E402


@pytest.fixture
def made(tmp_path):
    """A sheet written by the real sampler, plus its key and taxonomy."""
    tax, assign, clusters, posts = make_world(tmp_path)
    rows, summary = G.pick_gold(posts, assign, clusters, tax, generic_clusters=(4,))
    G.write_outputs(tmp_path / "gold", rows, summary, tax, R.load_rubrics(), params={})
    return {"tax": tax, "sheet": tmp_path / "gold" / "gold_labelling.xlsx", "key": tmp_path / "gold" / "gold_key.csv",
            "rows": rows, "dir": tmp_path}


def fill(path, edits):
    """edits = {(row_number, column_name): value}; saves the workbook."""
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb[G.SHEET_NAME]
    for (r, col), v in edits.items():
        ws.cell(row=r, column=G.SHEET_COLUMNS.index(col) + 1).value = v      # value=None in cell() would be ignored
    wb.save(path)


def read(made, **kw):
    return L.read_labels(made["sheet"], made["tax"], **kw)


# ------------------------------------------------------------------ the happy path

def test_blank_sheet_reads_as_thirty_empty_labels(made):
    rows = read(made)
    assert len(rows) == 30
    assert all(r[k] is None for r in rows for k in ("subtopic_id", "content_type", "difficulty", "alignment"))
    assert [r["gold_id"] for r in rows] == [r["gold_id"] for r in made["rows"]]


def test_activity_ids_are_exact(made):
    rows = read(made)
    assert [r["activity_id"] for r in rows] == [r["post"].activity_id for r in made["rows"]]
    assert all(len(str(r["activity_id"])) == 19 for r in rows)


def test_filled_labels_are_read(made):
    fill(made["sheet"], {(2, "my_subtopic"): "rag.chunking | RAG > Chunking", (2, "my_content_type"): "tutorial",
                         (2, "my_difficulty"): "advanced", (2, "my_alignment"): 7.5, (2, "my_cover_minutes"): 90,
                         (2, "my_understand_minutes"): 240, (2, "notes"): "  a  roadmap\n here "})
    r = read(made)[0]
    assert (r["subtopic_id"], r["content_type"], r["difficulty"]) == ("rag.chunking", "tutorial", "advanced")
    assert (r["alignment"], r["cover_minutes"], r["understand_minutes"]) == (7.5, 90, 240)
    assert r["notes"] == "a roadmap here"


def test_subtopic_can_be_typed_as_the_bare_id(made):
    fill(made["sheet"], {(2, "my_subtopic"): "careers.interviews"})
    assert read(made)[0]["subtopic_id"] == "careers.interviews"


def test_other_general_is_a_valid_label(made):
    fill(made["sheet"], {(2, "my_subtopic"): "other.general | Other / does not fit any subtopic"})
    assert read(made)[0]["subtopic_id"] == "other.general"


def test_blank_cells_are_allowed_and_stay_none(made):
    fill(made["sheet"], {(2, "my_content_type"): "news", (3, "my_difficulty"): "beginner", (4, "my_subtopic"): "   "})
    rows = read(made)
    assert rows[0]["content_type"] == "news" and rows[0]["difficulty"] is None
    assert rows[1]["difficulty"] == "beginner" and rows[1]["content_type"] is None
    assert rows[2]["subtopic_id"] is None                       # whitespace only counts as blank


def test_alignment_edges_are_valid(made):
    fill(made["sheet"], {(2, "my_alignment"): 0, (3, "my_alignment"): 10})
    rows = read(made)
    assert rows[0]["alignment"] == 0 and rows[1]["alignment"] == 10


def test_minutes_given_as_whole_floats_are_fine(made):
    fill(made["sheet"], {(2, "my_cover_minutes"): 60.0})
    assert read(made)[0]["cover_minutes"] == 60 and isinstance(read(made)[0]["cover_minutes"], int)


def test_every_difficulty_and_type_is_accepted(made):
    edits = {}
    for n, d in enumerate(R.DIFFICULTY_IDS, start=2):
        edits[(n, "my_difficulty")] = d
    for n, t in enumerate(R.CONTENT_TYPE_IDS, start=2):
        edits[(n, "my_content_type")] = t
    fill(made["sheet"], edits)
    rows = read(made)
    assert [r["difficulty"] for r in rows[:4]] == R.DIFFICULTY_IDS
    assert [r["content_type"] for r in rows[:6]] == R.CONTENT_TYPE_IDS


# ------------------------------------------------------------------ problems are all reported together

def test_bad_values_are_listed_together(made):
    fill(made["sheet"], {(2, "my_subtopic"): "rag.nonsense", (3, "my_content_type"): "blog",
                         (4, "my_difficulty"): "easy", (5, "my_alignment"): 11, (6, "my_alignment"): "high",
                         (7, "my_alignment"): -1, (8, "my_cover_minutes"): 0, (9, "my_cover_minutes"): 12.5,
                         (10, "my_understand_minutes"): "an hour", (11, "my_alignment"): True})
    with pytest.raises(L.LabelError) as err:
        read(made)
    msg = str(err.value)
    for part in ("my_subtopic 'rag.nonsense' is not on the list", "my_content_type 'blog' must be one of",
                 "my_difficulty 'easy' must be one of", "my_alignment '11'", "my_alignment 'high'",
                 "my_alignment '-1'", "my_cover_minutes '0'", "my_cover_minutes '12.5'",
                 "my_understand_minutes 'an hour'", "my_alignment 'True'"):
        assert part in msg
    assert msg.count("\n- ") == 10


def test_problem_names_the_row_and_gold_id(made):
    fill(made["sheet"], {(3, "my_difficulty"): "easy"})
    gid = made["rows"][1]["gold_id"]
    with pytest.raises(L.LabelError, match=f"row 3 \\({gid}\\)"):
        read(made)


def test_content_type_is_case_sensitive_to_the_list(made):
    fill(made["sheet"], {(2, "my_content_type"): "Tutorial"})
    with pytest.raises(L.LabelError, match="my_content_type 'Tutorial'"):
        read(made)


def test_minutes_of_one_are_valid_and_alignment_just_over_ten_is_not(made):
    fill(made["sheet"], {(2, "my_cover_minutes"): 1, (3, "my_alignment"): 10.01})
    with pytest.raises(L.LabelError) as err:
        read(made)
    assert "my_cover_minutes" not in str(err.value) and "10.01" in str(err.value)


# ------------------------------------------------------------------ structure of the workbook

def test_rounded_activity_id_is_caught(made):
    fill(made["sheet"], {(2, "activity_id"): 7.521e18})
    with pytest.raises(L.LabelError, match="rounded number"):
        read(made)


def test_missing_activity_id_is_caught(made):
    fill(made["sheet"], {(2, "activity_id"): None})
    with pytest.raises(L.LabelError, match="activity_id"):
        read(made)


def test_duplicate_gold_id_is_caught(made):
    fill(made["sheet"], {(3, "gold_id"): made["rows"][0]["gold_id"]})
    with pytest.raises(L.LabelError, match="appears twice"):
        read(made)


def test_empty_gold_id_is_caught(made):
    fill(made["sheet"], {(3, "gold_id"): None})
    with pytest.raises(L.LabelError, match="gold_id is empty"):
        read(made)


def test_missing_column_is_caught(made):
    from openpyxl import load_workbook
    wb = load_workbook(made["sheet"])
    wb[G.SHEET_NAME].cell(row=1, column=G.SHEET_COLUMNS.index("my_alignment") + 1, value="alignment")
    wb.save(made["sheet"])
    with pytest.raises(L.LabelError, match="missing columns.*my_alignment"):
        read(made)


def test_missing_sheet_is_caught(made):
    from openpyxl import load_workbook
    wb = load_workbook(made["sheet"])
    wb[G.SHEET_NAME].title = "Other"
    wb.save(made["sheet"])
    with pytest.raises(L.LabelError, match="no 'Label' sheet"):
        read(made)


def test_unreadable_file_is_caught(tmp_path, made):
    bad = tmp_path / "x.xlsx"
    bad.write_text("not a workbook", encoding="utf-8")
    with pytest.raises(L.LabelError, match="cannot read"):
        L.read_labels(bad, made["tax"])
    with pytest.raises(L.LabelError, match="cannot read"):
        L.read_labels(tmp_path / "missing.xlsx", made["tax"])


def test_trailing_empty_lines_are_ignored(made):
    from openpyxl import load_workbook
    wb = load_workbook(made["sheet"])
    ws = wb[G.SHEET_NAME]
    ws.cell(row=40, column=1, value=None)
    ws.cell(row=40, column=2, value="   ")
    wb.save(made["sheet"])
    assert len(read(made)) == 30


# ------------------------------------------------------------------ the key

def test_key_matches(made):
    assert len(read(made, key_ids=L.read_key(made["key"]))) == 30


def test_read_key_with_bom(made):
    key = L.read_key(made["key"])
    assert key[made["rows"][0]["gold_id"]] == made["rows"][0]["post"].activity_id


def test_wrong_activity_id_is_caught(made):
    fill(made["sheet"], {(2, "activity_id"): str(BASE + 999)})
    with pytest.raises(L.LabelError, match="differs from the key"):
        read(made, key_ids=L.read_key(made["key"]))


def test_unknown_gold_id_is_caught(made):
    fill(made["sheet"], {(2, "gold_id"): "g99"})
    with pytest.raises(L.LabelError, match="not in the key file"):
        read(made, key_ids=L.read_key(made["key"]))


def test_row_missing_from_the_sheet_is_caught(made):
    from openpyxl import load_workbook
    wb = load_workbook(made["sheet"])
    wb[G.SHEET_NAME].delete_rows(31)
    wb.save(made["sheet"])
    gid = made["rows"][-1]["gold_id"]
    with pytest.raises(L.LabelError, match=f"{gid} is in the key file but not in the sheet"):
        read(made, key_ids=L.read_key(made["key"]))


def test_key_is_optional(made):
    assert len(read(made, key_ids=None)) == 30


# ------------------------------------------------------------------ outputs

def test_coverage_counts(made):
    fill(made["sheet"], {(2, "my_subtopic"): "careers.interviews", (3, "my_subtopic"): "careers.interviews",
                         (2, "my_alignment"): 5, (4, "my_cover_minutes"): 30})
    assert L.coverage(read(made)) == {"subtopic_id": 2, "content_type": 0, "difficulty": 0, "alignment": 1,
                                      "cover_minutes": 1, "understand_minutes": 0}


def test_write_outputs(made):
    fill(made["sheet"], {(2, "my_subtopic"): "careers.interviews", (2, "my_alignment"): 0, (2, "notes"): "n"})
    rows = read(made)
    labels, ids = L.write_outputs(made["dir"] / "out", rows)
    with labels.open(encoding="utf-8-sig", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == L.CSV_COLUMNS and len(got) == 30
    assert got[0]["subtopic_id"] == "careers.interviews" and got[0]["alignment"] == "0.0" and got[0]["notes"] == "n"
    assert got[0]["content_type"] == "" and got[1]["alignment"] == ""      # blank stays blank, zero stays zero
    assert got[0]["activity_id"] == str(made["rows"][0]["post"].activity_id)
    parts = ids.read_text(encoding="utf-8").strip().split(",")
    assert parts == [str(r["post"].activity_id) for r in made["rows"]]


# ------------------------------------------------------------------ command line

def run(made, *extra):
    return L.main(["--sheet", str(made["sheet"]), "--taxonomy", str(made["dir"] / "t.yaml"),
                   "--out-dir", str(made["dir"] / "out"), *extra])


def test_main_reports_coverage_and_blanks(made, capsys):
    fill(made["sheet"], {(2, "my_subtopic"): "careers.interviews", (2, "my_content_type"): "news",
                         (2, "my_difficulty"): "beginner", (3, "my_subtopic"): "rag.chunking"})
    assert run(made, "--key", str(made["key"])) == 0
    out = capsys.readouterr().out
    assert "read 30 labelled posts" in out and "subtopic_id 2" in out and "content_type 1" in out
    assert made["rows"][1]["gold_id"] in out and "main labels still blank" in out
    assert (made["dir"] / "out" / L.LABEL_FILE).exists() and (made["dir"] / "out" / L.IDS_FILE).exists()


def test_main_no_note_when_all_main_labels_are_filled(made, capsys):
    edits = {}
    for n in range(2, 32):
        edits[(n, "my_subtopic")] = "rag.chunking"
        edits[(n, "my_content_type")] = "news"
        edits[(n, "my_difficulty")] = "beginner"
    fill(made["sheet"], edits)
    assert run(made) == 0
    assert "still blank" not in capsys.readouterr().out


def test_main_problems_exit_1_and_write_nothing(made, capsys):
    fill(made["sheet"], {(2, "my_difficulty"): "easy"})
    assert run(made) == 1
    assert "cannot read the labels" in capsys.readouterr().err
    assert not (made["dir"] / "out").exists()


def test_main_bad_key_path_exit_1(made, capsys):
    assert run(made, "--key", str(made["dir"] / "nope.csv")) == 1
    assert "cannot read the labels" in capsys.readouterr().err


def test_main_bad_taxonomy_exit_1(made, capsys):
    assert L.main(["--sheet", str(made["sheet"]), "--taxonomy", str(made["dir"] / "nope.yaml")]) == 1


# ------------------------------------------------------------------ extra edge cases

def test_numbers_typed_as_text_are_read(made):
    fill(made["sheet"], {(2, "my_alignment"): " 7.5 ", (2, "my_cover_minutes"): "90", (2, "my_understand_minutes"): "120.0"})
    r = read(made)[0]
    assert (r["alignment"], r["cover_minutes"], r["understand_minutes"]) == (7.5, 90, 120)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_numbers_are_a_clean_problem(made, value):
    fill(made["sheet"], {(2, "my_alignment"): value, (3, "my_cover_minutes"): value})
    with pytest.raises(L.LabelError) as err:
        read(made)
    assert str(err.value).count("\n- ") == 2


def test_type_and_difficulty_are_trimmed(made):
    fill(made["sheet"], {(2, "my_content_type"): " tutorial ", (2, "my_difficulty"): "beginner "})
    r = read(made)[0]
    assert (r["content_type"], r["difficulty"]) == ("tutorial", "beginner")


def test_excel_warnings_do_not_leak(made, monkeypatch):
    import warnings

    import openpyxl
    real = openpyxl.load_workbook

    def noisy(*a, **kw):
        warnings.warn("Data Validation extension is not supported and will be removed")
        return real(*a, **kw)

    monkeypatch.setattr(openpyxl, "load_workbook", noisy)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        read(made)
    assert seen == []


def test_empty_label_sheet_is_a_clean_problem(tmp_path, made):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = G.SHEET_NAME
    wb.save(tmp_path / "e.xlsx")
    with pytest.raises(L.LabelError, match="the sheet is empty"):
        L.read_labels(tmp_path / "e.xlsx", made["tax"])


def test_activity_id_stored_as_a_number_is_rejected(made):
    fill(made["sheet"], {(2, "activity_id"): 12345})               # a number cell, not text
    with pytest.raises(L.LabelError, match="rounded number"):
        read(made)


def test_main_notes_a_post_with_only_difficulty_blank(made, capsys):
    edits = {}
    for n in range(2, 32):
        edits[(n, "my_subtopic")] = "rag.chunking"
        edits[(n, "my_content_type")] = "news"
        edits[(n, "my_difficulty")] = "beginner"
    edits[(5, "my_difficulty")] = None
    fill(made["sheet"], edits)
    assert run(made) == 0
    out = capsys.readouterr().out
    assert made["rows"][3]["gold_id"] in out and "main labels still blank" in out


def test_excel_warnings_raised_while_reading_rows_do_not_leak(made, monkeypatch):
    """The real warning appears when openpyxl reads the rows, after the workbook is loaded."""
    import warnings

    import openpyxl
    real_load = openpyxl.load_workbook

    class NoisySheet:
        def __init__(self, sheet):
            self.sheet = sheet

        def iter_rows(self, *a, **kw):
            warnings.warn("Data Validation extension is not supported and will be removed")
            return self.sheet.iter_rows(*a, **kw)

    class NoisyBook:
        def __init__(self, wb):
            self.wb = wb
            self.sheetnames = wb.sheetnames

        def __getitem__(self, name):
            return NoisySheet(self.wb[name])

    monkeypatch.setattr(openpyxl, "load_workbook", lambda *a, **kw: NoisyBook(real_load(*a, **kw)))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert len(read(made)) == 30
    assert seen == []
