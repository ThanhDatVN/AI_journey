"""Tính toàn vẹn của khoá học: mọi chương có đủ thành phần, ngân hàng quiz đọc được, công cụ build hoạt động đúng."""

import json
import re
import sys
from pathlib import Path

import nbformat
import pytest

from ml_course.course import CHAPTERS, chapter_label, next_chapter
from ml_course.progress import record_result, render_progress, update_readme

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from generate_index import schedule_kind  # noqa: E402
from make_exercises import make_exercise  # noqa: E402


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_chapter_has_all_parts(chapter):
    d = ROOT / "notebooks" / chapter
    assert (d / "README.md").exists()
    assert (d / "90_exercises.ipynb").exists() and (d / "99_checkpoint.ipynb").exists()
    assert any(not p.name.startswith("9") for p in d.glob("*.ipynb")), "có ít nhất một bài học"
    assert (ROOT / "quizzes" / "chapter_quizzes" / f"{chapter}.json").exists()
    assert (ROOT / "solutions" / "checkpoints" / f"{chapter}.ipynb").exists()
    assert (ROOT / "solutions" / "exercises" / chapter / "90_exercises.ipynb").exists()


@pytest.mark.parametrize("path", sorted((ROOT / "quizzes").rglob("*.json")), ids=lambda p: p.stem)
def test_quiz_specs_are_well_formed(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["id"] == path.stem and spec["kind"] in {"Chapter", "Exam"}
    for section in spec["sections"].values():
        assert section["questions"], "mỗi phần có câu hỏi"
        for q in section["questions"]:
            assert q["answer_hashes"] and all(re.fullmatch(r"[0-9a-f]{20}", h) for h in q["answer_hashes"]), "đáp án chỉ lưu dạng băm"


def test_projects_exams_and_practice_present():
    for name in ["01_regression", "02_classification", "03_clustering", "04_time_series", "05_recommendation",
                 "06_nlp", "07_computer_vision", "08_end_to_end_ml_system"]:
        assert (ROOT / "projects" / name / "README.md").exists(), name
        assert (ROOT / "projects" / name / "project.ipynb").exists(), name
        assert (ROOT / "solutions" / "projects" / name / "project.ipynb").exists(), name
    assert len(list((ROOT / "quizzes" / "final_exams").glob("*.ipynb"))) == 6
    assert len(list((ROOT / "exercises").glob("*/*.ipynb"))) == 9


def test_learner_notebooks_have_no_solution_cells():
    for nb_path in list((ROOT / "projects").glob("*/project.ipynb")) + list((ROOT / "exercises").glob("*/*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        assert not any("solution" in c.metadata.get("tags", []) for c in nb.cells), nb_path


def test_course_order_helpers():
    assert CHAPTERS[0] == "00_course_orientation" and CHAPTERS[-1] == "41_capstone_projects" and len(CHAPTERS) == 42
    assert next_chapter("06_linear_regression") == "07_logistic_regression"
    assert next_chapter("41_capstone_projects") is None
    assert chapter_label("06_linear_regression") == "06 · Linear Regression"


def test_make_exercise_rewrites_paths_and_strips_code():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("[Colab](https://x/solutions/projects/01_regression/project.ipynb)"),
        nbformat.v4.new_code_cell("x = 1\n### START CODE HERE ### (≈ 1 dòng)\ny = x + 1\n### END CODE HERE ###"),
        nbformat.v4.new_markdown_cell("đáp án"),
    ]
    nb.cells[2].metadata["tags"] = ["solution"]
    out = make_exercise(nb, "01_regression", "solutions/projects", "projects")
    assert len(out.cells) == 2
    assert "projects/01_regression/project.ipynb" in out.cells[0].source and "solutions/" not in out.cells[0].source
    assert "y = None" in out.cells[1].source and "x + 1" not in out.cells[1].source
    assert out.cells[1].outputs == []


def test_schedule_kind_labels():
    assert schedule_kind("projects/01_regression/project.ipynb") == ("🛠️", "Project")
    assert schedule_kind("quizzes/final_exams/final_exam_1_foundations.ipynb") == ("🎓", "Thi")
    assert schedule_kind("exercises/basic/01_python_numpy_pandas.ipynb")[1] == "Luyện tập"
    assert schedule_kind("notebooks/06_linear_regression/99_checkpoint.ipynb")[1] == "Quiz"
    assert schedule_kind("notebooks/06_linear_regression/91_assignment_x.ipynb")[1] == "Assignment"
    assert schedule_kind("notebooks/06_linear_regression/01_x.ipynb") == ("📖", None)


def test_progress_record_and_readme(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "quizzes").mkdir()
    monkeypatch.setenv("ML_COURSE_ROOT", str(tmp_path))
    record_result("00_course_orientation", {"status": "REVIEW", "components": {}})
    record_result("00_course_orientation", {"status": "PASS", "components": {}})
    data = json.loads((tmp_path / "progress" / "progress.json").read_text(encoding="utf-8"))
    entry = data["chapters"]["00_course_orientation"]
    assert entry["passed"] and len(entry["attempts"]) == 2 and entry["latest"]["status"] == "PASS"
    text = render_progress()
    assert "(1/2)" in text and "01 · Python for ML" in text            # 1/2 chương Foundations, chương tiếp theo
    readme = tmp_path / "README.md"
    readme.write_text("trước\n<!-- PROGRESS:START -->\ncũ\n<!-- PROGRESS:END -->\nsau\n", encoding="utf-8")
    assert update_readme(readme)
    updated = readme.read_text(encoding="utf-8")
    assert "cũ" not in updated and "TOTAL" in updated and updated.startswith("trước") and updated.rstrip().endswith("sau")
