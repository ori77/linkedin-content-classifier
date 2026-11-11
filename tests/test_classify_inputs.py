"""Tests for pipeline/classify_inputs.py (Step 9): taxonomy, assignments and posts loaders."""
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import classify_inputs as I  # noqa: E402

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "Retrieval-Augmented Generation", "description": "RAG systems.",
         "subtopics": [
             {"id": "rag.chunking", "name": "Chunking", "description": "How to split\ndocuments.",
              "keywords": ["chunk", "split"]},
             {"id": "rag.evaluation", "name": "RAG Evaluation", "description": "Measuring RAG."},
         ]},
        {"id": "careers", "name": "Career Development", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
    ],
    "other": {"id": "other.general", "name": "Other / General", "description": "Fits nothing."},
}


@pytest.fixture
def tax_data():
    return copy.deepcopy(TAXONOMY)


def write_yaml(tmp_path, data, name="tax.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def problems_with(data, fragment):
    return [p for p in I.validate_taxonomy(data) if fragment in p]


# ------------------------------------------------------------ taxonomy

def test_valid_taxonomy_has_no_problems(tax_data):
    assert I.validate_taxonomy(tax_data) == []


def test_load_taxonomy_builds_subtopics_in_file_order(tmp_path):
    tax = I.load_taxonomy(write_yaml(tmp_path, TAXONOMY))
    assert [s.id for s in tax.subtopics] == ["rag.chunking", "rag.evaluation", "careers.interviews"]
    assert tax.choice_ids == ["rag.chunking", "rag.evaluation", "careers.interviews", "other.general"]
    assert tax.version == "v1" and len(tax.sha) == 64


def test_subtopic_fields_and_label(tmp_path):
    s = I.load_taxonomy(write_yaml(tmp_path, TAXONOMY)).by_id["rag.chunking"]
    assert s.topic_id == "rag" and s.topic_name == "Retrieval-Augmented Generation"
    assert s.label == "Retrieval-Augmented Generation > Chunking"
    assert s.description == "How to split documents."          # line breaks folded into spaces
    assert s.keywords == ("chunk", "split")


def test_topic_of(tmp_path):
    tax = I.load_taxonomy(write_yaml(tmp_path, TAXONOMY))
    assert tax.topic_of("rag.evaluation") == "rag"
    assert tax.topic_of("other.general") == "other"
    with pytest.raises(KeyError):
        tax.topic_of("nonsense")


def test_taxonomy_sha_ignores_windows_line_endings(tmp_path):
    # write_text turns \n into \r\n on Windows, so build both files from explicit bytes
    lf = yaml.safe_dump(TAXONOMY).encode("utf-8").replace(b"\r\n", b"\n")
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_bytes(lf)
    b.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert b"\r\n" in b.read_bytes() and b"\r" not in a.read_bytes()
    assert I.load_taxonomy(a).sha == I.load_taxonomy(b).sha


def test_taxonomy_sha_changes_with_content(tmp_path, tax_data):
    a = I.load_taxonomy(write_yaml(tmp_path, tax_data, "a.yaml"))
    tax_data["topics"][0]["subtopics"][0]["description"] = "Changed."
    b = I.load_taxonomy(write_yaml(tmp_path, tax_data, "b.yaml"))
    assert a.sha != b.sha


def test_duplicate_subtopic_id(tax_data):
    tax_data["topics"][0]["subtopics"][1]["id"] = "rag.chunking"
    assert problems_with(tax_data, "used twice")


def test_duplicate_topic_id(tax_data):
    tax_data["topics"][1]["id"] = "rag"
    assert problems_with(tax_data, "topic id 'rag' is used twice")


def test_subtopic_must_start_with_topic_id(tax_data):
    tax_data["topics"][0]["subtopics"][0]["id"] = "careers.chunking"
    assert problems_with(tax_data, "must start with its topic id")


def test_topic_id_other_is_reserved(tax_data):
    tax_data["topics"][1]["id"] = "other"
    tax_data["topics"][1]["subtopics"][0]["id"] = "other.interviews"
    assert problems_with(tax_data, "reserved")


def test_other_entry_required(tax_data):
    del tax_data["other"]
    assert problems_with(tax_data, "'other' must exist")


def test_other_must_have_the_right_id(tax_data):
    tax_data["other"]["id"] = "misc"
    assert problems_with(tax_data, "'other' must exist")


@pytest.mark.parametrize("path", [("topics", 0, "name"), ("topics", 0, "description")])
def test_empty_topic_text(tax_data, path):
    tax_data["topics"][path[1]][path[2]] = " "
    assert problems_with(tax_data, "is empty")


@pytest.mark.parametrize("field", ["name", "description"])
def test_empty_subtopic_text(tax_data, field):
    tax_data["topics"][0]["subtopics"][0][field] = ""
    assert problems_with(tax_data, "is empty")


def test_topic_without_subtopics(tax_data):
    tax_data["topics"][1]["subtopics"] = []
    assert problems_with(tax_data, "at least one subtopic")


def test_keywords_must_be_text(tax_data):
    tax_data["topics"][0]["subtopics"][0]["keywords"] = [1, 2]
    assert problems_with(tax_data, "'keywords'")


def test_missing_version(tax_data):
    tax_data["version"] = ""
    assert problems_with(tax_data, "'version'")


def test_no_topics(tax_data):
    tax_data["topics"] = []
    assert problems_with(tax_data, "'topics'")


def test_top_level_must_be_mapping():
    assert I.validate_taxonomy([1, 2])


def test_load_taxonomy_lists_all_problems(tmp_path, tax_data):
    tax_data["version"] = ""
    tax_data["other"]["id"] = "misc"
    with pytest.raises(I.InputError) as e:
        I.load_taxonomy(write_yaml(tmp_path, tax_data))
    assert "'version'" in str(e.value) and "'other' must exist" in str(e.value)


def test_load_taxonomy_missing_file(tmp_path):
    with pytest.raises(I.InputError, match="cannot read"):
        I.load_taxonomy(tmp_path / "nope.yaml")


def test_load_taxonomy_bad_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(I.InputError, match="cannot read"):
        I.load_taxonomy(p)


# ------------------------------------------------------------ assignments

@pytest.fixture
def tax(tmp_path):
    return I.load_taxonomy(write_yaml(tmp_path, TAXONOMY))


def write_csv(tmp_path, text, name="a.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


GOOD_ASSIGN = ("activity_id,cluster,topic_id,subtopic_id\n"
               "7521561188925616128,-1,other,other.general\n"
               "7521877444732116992,3,rag,rag.chunking\n")


def test_assignments_load(tmp_path, tax):
    got = I.load_assignments(write_csv(tmp_path, GOOD_ASSIGN), tax)
    assert got == {7521561188925616128: "other.general", 7521877444732116992: "rag.chunking"}


def test_assignments_large_ids_stay_exact(tmp_path, tax):
    got = I.load_assignments(write_csv(tmp_path, GOOD_ASSIGN), tax)
    assert 7521877444732116992 in got                       # no float rounding of 19-digit ids


def test_assignments_unknown_subtopic(tmp_path, tax):
    bad = GOOD_ASSIGN + "7521877444732116993,3,rag,rag.nonsense\n"
    with pytest.raises(I.InputError, match="not in the taxonomy"):
        I.load_assignments(write_csv(tmp_path, bad), tax)


def test_assignments_duplicate_id(tmp_path, tax):
    bad = GOOD_ASSIGN + "7521877444732116992,3,rag,rag.chunking\n"
    with pytest.raises(I.InputError, match="appears twice"):
        I.load_assignments(write_csv(tmp_path, bad), tax)


def test_assignments_bad_id(tmp_path, tax):
    bad = GOOD_ASSIGN + "abc,3,rag,rag.chunking\n"
    with pytest.raises(I.InputError, match="not a whole number"):
        I.load_assignments(write_csv(tmp_path, bad), tax)


def test_assignments_missing_column(tmp_path, tax):
    with pytest.raises(I.InputError, match="missing columns"):
        I.load_assignments(write_csv(tmp_path, "activity_id,cluster\n1,2\n"), tax)


def test_assignments_missing_file(tmp_path, tax):
    with pytest.raises(I.InputError, match="cannot read"):
        I.load_assignments(tmp_path / "nope.csv", tax)


def test_assignments_report_all_problems(tmp_path, tax):
    bad = GOOD_ASSIGN + "x,3,rag,rag.chunking\n7521877444732116999,3,rag,zzz\n"
    with pytest.raises(I.InputError) as e:
        I.load_assignments(write_csv(tmp_path, bad), tax)
    assert "not a whole number" in str(e.value) and "not in the taxonomy" in str(e.value)


# ------------------------------------------------------------ posts

POSTS_HEADER = "activity_id,text,text_clean,attachment_title,attachment_site,urls,word_count\n"


def test_load_posts_basic(tmp_path):
    csv_text = POSTS_HEADER + ('1,raw one,Clean one,A title,example.com,https://example.com/x,2\n'
                               '2,raw two,Clean two,,,,2\n')
    posts, skipped = I.load_posts(write_csv(tmp_path, csv_text))
    assert [p.activity_id for p in posts] == [1, 2] and skipped == []
    assert posts[0].text == "Clean one" and posts[0].attachment_title == "A title"
    assert posts[0].attachment_site == "example.com" and posts[0].links == "https://example.com/x"
    assert posts[1].attachment_title == ""


def test_load_posts_word_count_is_computed_from_the_text(tmp_path):
    csv_text = POSTS_HEADER + "1,x,one two three four,,,,999\n"       # the csv column is ignored
    posts, _ = I.load_posts(write_csv(tmp_path, csv_text))
    assert posts[0].word_count == 4


def test_load_posts_falls_back_to_raw_text(tmp_path):
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + "1,raw text only,,,,,\n"))
    assert posts[0].text == "raw text only"


def test_load_posts_skips_rows_without_text(tmp_path):
    csv_text = POSTS_HEADER + "1,,,,,,\n2,,  ,,,,\n3,hello,hello,,,,1\n"
    posts, skipped = I.load_posts(write_csv(tmp_path, csv_text))
    assert [p.activity_id for p in posts] == [3]
    assert skipped == [(1, "no text"), (2, "no text")]


def test_load_posts_treats_nan_text_as_empty(tmp_path):
    posts, skipped = I.load_posts(write_csv(tmp_path, POSTS_HEADER + "1,nan,nan,,,,\n"))
    assert posts == [] and skipped == [(1, "no text")]


def test_load_posts_keeps_big_ids_exact(tmp_path):
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + "7521561188925616128,a,a,,,,1\n"))
    assert posts[0].activity_id == 7521561188925616128


def test_load_posts_duplicate_ids(tmp_path):
    with pytest.raises(I.InputError, match="appears twice"):
        I.load_posts(write_csv(tmp_path, POSTS_HEADER + "1,a,a,,,,1\n1,b,b,,,,1\n"))


def test_load_posts_bad_id(tmp_path):
    with pytest.raises(I.InputError, match="not a whole number"):
        I.load_posts(write_csv(tmp_path, POSTS_HEADER + "abc,a,a,,,,1\n"))


def test_load_posts_missing_columns(tmp_path):
    with pytest.raises(I.InputError, match="missing column 'text_clean'"):
        I.load_posts(write_csv(tmp_path, "activity_id,text\n1,a\n"))


def test_load_posts_missing_file(tmp_path):
    with pytest.raises(I.InputError, match="cannot read"):
        I.load_posts(tmp_path / "nope.csv")


def test_long_posts_are_cut_and_flagged(tmp_path):
    long_text = "word " * 5000                                       # 25,000 characters
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + f"1,x,{long_text.strip()},,,,5000\n"))
    assert posts[0].truncated is True and len(posts[0].text) == I.MAX_POST_CHARS


def test_exactly_the_limit_is_not_truncated(tmp_path):
    text = "a" * I.MAX_POST_CHARS
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + f"1,x,{text},,,,1\n"))
    assert posts[0].truncated is False and len(posts[0].text) == I.MAX_POST_CHARS
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + f"1,x,{text}b,,,,1\n"))
    assert posts[0].truncated is True


def test_posts_csv_with_a_byte_order_mark(tmp_path):
    p = tmp_path / "posts.csv"
    p.write_bytes(b"\xef\xbb\xbf" + (POSTS_HEADER + "1,x,hello world,,,,1\n").encode("utf-8"))
    posts, _ = I.load_posts(p)
    assert [x.activity_id for x in posts] == [1]


def test_taxonomy_file_with_a_byte_order_mark(tmp_path):
    p = tmp_path / "tax.yaml"
    p.write_bytes(b"\xef\xbb\xbf" + yaml.safe_dump(TAXONOMY).encode("utf-8"))
    assert I.load_taxonomy(p).version == TAXONOMY["version"]


def test_assignments_csv_with_a_byte_order_mark_and_windows_line_endings(tmp_path, tax):
    sid = tax.subtopics[0].id
    p = tmp_path / "a.csv"
    p.write_bytes(b"\xef\xbb\xbf" + f"activity_id,cluster,topic_id,subtopic_id\r\n5,3,t,{sid}\r\n".encode("utf-8"))
    assert I.load_assignments(p, tax) == {5: sid}


def test_short_posts_are_not_flagged(tmp_path):
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + "1,x,short,,,,1\n"))
    assert posts[0].truncated is False


def test_links_are_cut_to_300_characters(tmp_path):
    posts, _ = I.load_posts(write_csv(tmp_path, POSTS_HEADER + "1,x,t,,," + "u" * 500 + ",1\n"))
    assert len(posts[0].links) == 300
