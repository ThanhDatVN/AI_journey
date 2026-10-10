"""Test engine checkpoint + công cụ sinh bài tập (không phụ thuộc file quiz thật)."""

import json
import sys
from pathlib import Path

import pytest

from ml_course import checkpoint as ckp
from ml_course.checks import check_close, check_less
from ml_course.progress import bar, render_progress

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nbutils import strip_solution_code  # noqa: E402


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    qdir = tmp_path / "quizzes" / "chapter_quizzes"
    qdir.mkdir(parents=True)
    sid = "06_linear_regression"

    def q(qid, qtype, answer, **kw):
        norm = ckp.normalize_answer(answer, qtype, kw.get("decimals"))[0]
        return {"id": qid, "type": qtype, "topic": f"topic {qid}", "answer_hashes": [ckp.hash_answer(sid, qid, norm)], **kw}

    spec = {
        "id": sid, "kind": "Chapter", "title": "06 · Linear Regression", "next": "07_logistic_regression",
        "sections": {
            "theory": {"label": "Theory Quiz", "threshold": 0.8,
                       "questions": [q("t01", "mcq", "b"), q("t02", "tf", True), q("t03", "multi", "ac")]},
            "math": {"label": "Math Quiz", "threshold": 0.8, "questions": [q("m01", "numeric", 0.971, decimals=2)]},
        },
        "tasks": {"coding": {"label": "Coding Exercise", "ids": ["c01"]}},
        "self_assess": {"analysis": {"label": "Model Analysis", "threshold": 0.8}},
    }
    (qdir / f"{sid}.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setenv("ML_COURSE_ROOT", str(tmp_path))
    return tmp_path


def test_normalize_answer_variants():
    assert ckp.normalize_answer("B)", "mcq") == ["b"]
    assert ckp.normalize_answer("Đúng", "tf") == ["true"]
    assert ckp.normalize_answer(["c", "a"], "multi") == ["ac"]
    assert ckp.normalize_answer(0.9709, "numeric", 2)[0] == "0.97"
    assert ckp.normalize_answer(-0.001, "numeric", 2)[0] == "0.00"
    assert ckp.normalize_answer(None, "mcq") == []


def test_full_checkpoint_pass(fake_repo, capsys):
    cp = ckp.Checkpoint("06_linear_regression")
    assert cp.grade("theory", {"t01": "b", "t02": "đúng", "t03": "c,a"}).passed
    assert cp.grade("math", {"m01": 0.9712}).passed
    with cp.task("c01"):
        check_close(1.0, 1.0)
    cp.self_assess("analysis", {"a": True, "b": True, "c": True, "d": True, "e": False})
    cp.report()
    assert cp.status() == "PASS"
    out = capsys.readouterr().out
    assert "STATUS: ✅ PASS" in out and "07 · Logistic Regression" in out
    saved = json.loads((fake_repo / "progress" / "progress.json").read_text(encoding="utf-8"))
    assert saved["chapters"]["06_linear_regression"]["passed"] is True


def test_checkpoint_review_lists_weak_topics(fake_repo, capsys):
    cp = ckp.Checkpoint("06_linear_regression", record=False)
    cp.grade("theory", {"t01": "a", "t02": True, "t03": "ac"})
    cp.grade("math", {"m01": 0.5})
    with cp.task("c01"):
        check_less(5, 1)
    cp.self_assess("analysis", {"a": True})
    cp.report()
    assert cp.status() == "REVIEW"
    out = capsys.readouterr().out
    assert "Weak topics" in out and "topic t01" in out and "topic m01" in out


def test_incomplete_when_component_missing(fake_repo):
    cp = ckp.Checkpoint("06_linear_regression", record=False)
    cp.grade("theory", {"t01": "b", "t02": True, "t03": "ac"})
    assert cp.status() == "INCOMPLETE"


def test_strip_solution_code_assignments_and_starter():
    src = (
        "def gini(y):\n"
        "    ### START CODE HERE ### (≈ 2 dòng)\n"
        "    p = np.bincount(y) / len(y)\n"
        "    result = 1 - np.sum(p ** 2)\n"
        "    ### END CODE HERE ###\n"
        "    return result\n"
        "### START CODE HERE ###\n"
        "# STARTER: model = ...\n"
        "model = Fancy().fit(X)\n"
        "### END CODE HERE ###"
    )
    out = strip_solution_code(src)
    assert "p = None" in out and "result = None" in out and "np.bincount" not in out
    assert "model = ..." in out and "Fancy" not in out
    assert out.count("### START CODE HERE ###") == 2


def test_strip_solution_code_tuples_with_blocks_and_multiline():
    src = (
        "### START CODE HERE ###\n"
        "X_tr, X_te = split(X)\n"
        "_, counts = np.unique(y, return_counts=True)\n"
        "model = fit(X_tr,\n"
        "            lr=0.1)\n"
        "with torch.no_grad():\n"
        "    acc = model(X_te).mean()\n"
        "for k in range(3):\n"
        "    tmp = k\n"
        "### END CODE HERE ###"
    )
    out = strip_solution_code(src)
    stub = out.split("\n")[1:-1]
    assert stub == ["X_tr, X_te = None, None", "counts = None", "model = None", "acc = None"]
    compile(out, "<stub>", "exec")


def test_strip_solution_code_comment_only_starter_stays_valid_python():
    src = (
        "def predict(node, x):\n"
        "    ### START CODE HERE ###\n"
        "    # STARTER: # đi xuống trái/phải tới khi gặp lá\n"
        "    while 'leaf' not in node:\n"
        "        node = node['left']\n"
        "    return node['leaf']\n"
        "    ### END CODE HERE ###\n"
        "\n"
        "\n"
        "for epoch in range(3):\n"
        "    ### START CODE HERE ###\n"
        "    # STARTER: # một bước SGD\n"
        "    step()\n"
        "    ### END CODE HERE ###"
    )
    out = strip_solution_code(src)
    assert out.count("    pass") == 2 and "node['left']" not in out and "step()" not in out
    compile(out, "<stub>", "exec")


def test_progress_rendering():
    assert bar(0.5, 10) == "█████░░░░░"
    text = render_progress({"chapters": {"00_course_orientation": {"passed": True}}})
    assert "Foundations" in text and "NEXT: 01 · Python for ML" in text
