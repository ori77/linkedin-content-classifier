"""Tests for pipeline/step9_classify.py (Step 9): the command line, resume, failures.

Most tests run the real script against a tiny fake Ollama server on a local port, so the real
HTTP code in ollama_json.py is exercised too.
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import step9_classify as S  # noqa: E402

TAXONOMY = {
    "version": "v1",
    "topics": [
        {"id": "rag", "name": "Retrieval-Augmented Generation", "description": "RAG systems.",
         "subtopics": [{"id": "rag.chunking", "name": "Chunking", "description": "Splitting documents."}]},
        {"id": "careers", "name": "Career Development", "description": "Jobs.",
         "subtopics": [{"id": "careers.interviews", "name": "Interviews", "description": "Prep."}]},
    ],
    "other": {"id": "other.general", "name": "Other / General", "description": "Fits nothing."},
}
SUMMARY = " ".join(["word"] * 60)
ANSWER_B = {"key_points": ["a point"], "stated_size": "", "stated_count": 0, "stated_unit": "none",
            "basis": "post_only", "content_type": "tutorial",
            "content_type_reason": "r", "difficulty": "intermediate", "difficulty_reason": "r",
            "cover_minutes": 2, "understand_minutes": 10, "time_reason": "r",
            "alignment": {"applicability": 8, "technical_depth": 6, "hands_on": 9, "currency": 7,
                          "career_value": 5, "production_relevance": 4},
            "alignment_reason": "r", "summary": SUMMARY, "self_confidence": 0.8}
ANSWER_A = {"evidence": "chunking", "subtopic_id": "rag.chunking"}

POSTS_CSV = (
    "activity_id,text,text_clean,attachment_title,attachment_site,urls,word_count\n"
    "1,x,Ten words about chunking documents for retrieval systems in practice.,,,,10\n"
    "2,x,A second post about interview preparation for AI engineers.,,,,9\n"
    "3,,,,,,\n"
    "4,x,Ten words about chunking documents for retrieval systems in practice.,,,,10\n"
)
ASSIGN_CSV = ("activity_id,cluster,topic_id,subtopic_id\n1,0,rag,rag.chunking\n2,1,careers,careers.interviews\n"
              "3,-1,other,other.general\n4,0,rag,rag.chunking\n")


class FakeOllama(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests = []
        self.mode = "ok"                    # ok | missing_model
        self.fail_marker = None             # posts containing this text get unusable answers

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(payload)
        if self.server.mode == "missing_model":
            self.send_response(404)
            self.end_headers()
            return
        is_a = "evidence" in payload["format"]["properties"]
        user = payload["messages"][1]["content"]
        if self.server.fail_marker and self.server.fail_marker in user:
            content = "this is not json"
        else:
            content = json.dumps(ANSWER_A if is_a else ANSWER_B)
        body = json.dumps({"message": {"content": content}, "done_reason": "stop",
                           "prompt_eval_count": 50, "eval_count": 10}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = FakeOllama()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def files(tmp_path):
    (tmp_path / "tax.yaml").write_text(yaml.safe_dump(TAXONOMY), encoding="utf-8")
    (tmp_path / "posts.csv").write_text(POSTS_CSV, encoding="utf-8")
    (tmp_path / "assign.csv").write_text(ASSIGN_CSV, encoding="utf-8")
    return tmp_path


def argv(files, server, *extra, out="out.jsonl"):
    return ["--csv", str(files / "posts.csv"), "--taxonomy", str(files / "tax.yaml"),
            "--assignments", str(files / "assign.csv"), "--model", "m", "--url", server.url,
            "--out", str(files / out), *extra]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------ end to end

def test_full_run(files, server, capsys):
    assert S.main(argv(files, server)) == 0
    recs = rows(files / "out.jsonl")
    assert [r["activity_id"] for r in recs] == [1, 2, 4]               # post 3 has no text; 4 is a repost of 1
    assert all(r["created_at"] and r["cache_key"] for r in recs)
    assert len({r["cache_key"] for r in recs}) == 3                    # the repost has its own row
    assert recs[0]["subtopic_id"] == "rag.chunking" and recs[0]["alignment_score"] == 7.0
    assert len(server.requests) == 6                                    # two calls per post
    out = capsys.readouterr().out
    assert "3 posts with text (1 without text skipped)" in out and "Done: 3 classified" in out
    assert "Time basis: stated_in_post 0, estimated_from_post 0, post_only 3" in out


def test_requests_carry_the_settings(files, server):
    S.main(argv(files, server, "--limit", "1"))
    first = server.requests[0]
    assert first["model"] == "m" and first["think"] is False and first["stream"] is False
    assert first["options"]["temperature"] == 0 and first["options"]["num_ctx"] == 8192
    assert "presence_penalty" not in first["options"]
    assert first["messages"][0]["role"] == "system" and "SUBTOPICS" in first["messages"][0]["content"]


def test_presence_penalty_think_and_num_ctx_options(files, server):
    S.main(argv(files, server, "--limit", "1", "--presence-penalty", "0", "--think", "--num-ctx", "4096"))
    first = server.requests[0]
    assert first["options"]["presence_penalty"] == 0.0 and first["think"] is True
    assert first["options"]["num_ctx"] == 4096


def test_resume_does_not_redo_work(files, server, capsys):
    S.main(argv(files, server))
    before = (files / "out.jsonl").read_text(encoding="utf-8")
    server.requests.clear()
    assert S.main(argv(files, server)) == 0
    assert server.requests == [] and (files / "out.jsonl").read_text(encoding="utf-8") == before
    assert "3 already done" in capsys.readouterr().out


def test_limit_then_continue(files, server):
    S.main(argv(files, server, "--limit", "1"))
    assert [r["activity_id"] for r in rows(files / "out.jsonl")] == [1]
    S.main(argv(files, server))
    assert [r["activity_id"] for r in rows(files / "out.jsonl")] == [1, 2, 4]


def test_ids_option(files, server):
    S.main(argv(files, server, "--ids", "2"))
    assert [r["activity_id"] for r in rows(files / "out.jsonl")] == [2]


def test_unknown_id_is_an_error(files, server, capsys):
    assert S.main(argv(files, server, "--ids", "2,999")) == 2
    assert "999" in capsys.readouterr().out and server.requests == []


def test_bad_id_text_is_an_error(files, server, capsys):
    assert S.main(argv(files, server, "--ids", "abc")) == 2
    assert "not a whole number" in capsys.readouterr().out


def test_changing_the_model_redoes_the_posts(files, server):
    S.main(argv(files, server, "--limit", "1"))
    args = argv(files, server, "--limit", "1")
    args[args.index("--model") + 1] = "other-model"
    S.main(args)
    recs = rows(files / "out.jsonl")
    assert [r["model"] for r in recs] == ["m", "other-model"] and recs[0]["cache_key"] != recs[1]["cache_key"]


def test_a_failed_post_is_recorded_and_the_run_continues(files, server, capsys):
    server.fail_marker = "interview preparation"
    assert S.main(argv(files, server)) == 1
    assert [r["activity_id"] for r in rows(files / "out.jsonl")] == [1, 4]
    failures = rows(files / "out.jsonl.failures.jsonl")
    assert len(failures) == 1 and failures[0]["activity_id"] == 2 and failures[0]["model"] == "m"
    assert "not valid JSON" in failures[0]["error"]
    out = capsys.readouterr().out
    assert "FAILED" in out and "1 failed" in out and "out.jsonl.failures.jsonl" in out


def test_failed_posts_are_retried_on_the_next_run(files, server):
    server.fail_marker = "interview preparation"
    S.main(argv(files, server))
    server.fail_marker = None
    assert S.main(argv(files, server)) == 0
    assert [r["activity_id"] for r in rows(files / "out.jsonl")] == [1, 4, 2]


def test_missing_model_stops_the_run(files, server, capsys):
    server.mode = "missing_model"
    assert S.main(argv(files, server)) == 2
    assert not (files / "out.jsonl").exists()
    assert "ollama pull m" in capsys.readouterr().out


def test_ollama_not_running(files, capsys):
    args = ["--csv", str(files / "posts.csv"), "--taxonomy", str(files / "tax.yaml"),
            "--assignments", str(files / "assign.csv"), "--model", "m", "--out", str(files / "o.jsonl"),
            "--url", "http://127.0.0.1:9"]
    assert S.main(args) == 2
    assert "Cannot reach Ollama" in capsys.readouterr().out


def test_show_prompt_does_not_call_the_model(files, server, capsys):
    assert S.main(argv(files, server, "--show-prompt")) == 0
    out = capsys.readouterr().out
    for title in ("SYSTEM PROMPT, CALL A", "USER PROMPT, CALL A", "SYSTEM PROMPT, CALL B", "USER PROMPT, CALL B"):
        assert title in out
    assert "Ten words about chunking" in out and "tokens of the 8192 available" in out
    assert server.requests == [] and not (files / "out.jsonl").exists()


def test_crash_damage_in_the_output_file_is_survived(files, server):
    (files / "out.jsonl").write_text('{"cache_key": "abc", "activity_id": 77}\n{"activity_id": 5, "cache_ke', encoding="utf-8")
    assert S.main(argv(files, server)) == 0
    lines = (files / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 + 3                                   # old good line, torn line, three new lines
    assert [json.loads(x)["activity_id"] for x in lines[-3:]] == [1, 2, 4]


# ------------------------------------------------------------ bad inputs

@pytest.mark.parametrize("missing", ["posts.csv", "tax.yaml", "assign.csv"])
def test_missing_input_files_stop_the_run(files, server, capsys, missing):
    (files / missing).unlink()
    assert S.main(argv(files, server)) == 2
    assert "cannot read" in capsys.readouterr().out and server.requests == []


def test_bad_rubrics_stop_the_run(files, server, capsys):
    bad = files / "rub.yaml"
    bad.write_text("version: v1\n", encoding="utf-8")
    assert S.main(argv(files, server, "--rubrics", str(bad))) == 2
    assert "rubrics problems" in capsys.readouterr().out


def test_assignment_for_unknown_subtopic_stops_the_run(files, server, capsys):
    (files / "assign.csv").write_text(ASSIGN_CSV + "5,0,rag,rag.nonsense\n", encoding="utf-8")
    assert S.main(argv(files, server)) == 2
    assert "not in the taxonomy" in capsys.readouterr().out


# ------------------------------------------------------------ small helpers

def test_default_out_name():
    assert S.default_out("qwen3.5:4b") == "classifications_qwen3.5-4b.jsonl"
    assert S.default_out("hf.co/user/model:Q4") == "classifications_hf.co-user-model-Q4.jsonl"


def test_read_done_keys_ignores_garbage(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"cache_key": "a"}\nnot json\n{"no_key": 1}\n[1]\n{"cache_key": "b"}\n{"cache_ke', encoding="utf-8")
    assert S.read_done_keys(p) == {"a", "b"}
    assert S.read_done_keys(tmp_path / "missing.jsonl") == set()


def test_append_line_repairs_a_missing_newline(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}', encoding="utf-8")
    S.append_line(p, {"b": 2})
    assert p.read_text(encoding="utf-8") == '{"a": 1}\n{"b": 2}\n'


def test_append_line_creates_the_file(tmp_path):
    p = tmp_path / "new.jsonl"
    S.append_line(p, {"ü": "naïve"})
    assert p.read_text(encoding="utf-8") == '{"ü": "naïve"}\n'          # not escaped


# ------------------------------------------------------------ the script itself

def test_script_runs_from_the_command_line(tmp_path):
    script = ROOT / "pipeline" / "step9_classify.py"
    assert subprocess.run([sys.executable, str(script), "--help"], capture_output=True).returncode == 0
    r = subprocess.run([sys.executable, str(script), "--csv", str(tmp_path / "no.csv"),
                        "--taxonomy", str(tmp_path / "no.yaml"), "--assignments", str(tmp_path / "no2.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "cannot read" in r.stdout
