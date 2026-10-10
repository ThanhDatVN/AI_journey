"""Build checkpoint & bài thi từ file YAML (nguồn có đáp án, nằm trong solutions/).

    solutions/quizzes/<chapter>.yaml  →  quizzes/chapter_quizzes/<chapter>.json   (đáp án đã hash)
                                       →  notebooks/<chapter>/99_checkpoint.ipynb  (cho người học)
                                       →  solutions/checkpoints/<chapter>.ipynb    (bản điền sẵn — phải ra PASS)
    solutions/exams/<exam>.yaml       →  quizzes/final_exams/<exam>.json + .ipynb
                                       →  solutions/checkpoints/<exam>.ipynb

Dùng:
    python tools/build_quizzes.py            # build tất cả
    python tools/build_quizzes.py 06 07      # chỉ các file bắt đầu bằng "06", "07"
    python tools/build_quizzes.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbutils import (
    ROOT,
    colab_badge,
    finalize,
    new_notebook,
    notebook_text,
    remove_starter_comments,
    setup_cell_source,
    strip_solution_code,
    write_if_changed,
)

sys.path.insert(0, str(ROOT / "src"))
from ml_course.checkpoint import hash_answer, normalize_answer  # noqa: E402
from ml_course.course import chapter_label, next_chapter  # noqa: E402

DEFAULT_LABELS = {
    "theory": "Theory Quiz",
    "math": "Math Quiz",
    "coding": "Coding Exercise",
    "analysis": "Model Analysis",
    "project": "Mini Project",
}
ANSWER_HINT = {
    "mcq": "a/b/c/d",
    "multi": 'chọn nhiều, vd "ac"',
    "tf": "True/False",
    "numeric": "số",
    "text": "từ/cụm từ",
}


# ----------------------------------------------------------------------------- spec (JSON)
def build_spec(src: dict, spec_id: str, kind: str) -> dict:
    spec = {
        "id": spec_id,
        "kind": kind,
        "title": src.get("title") or chapter_label(spec_id),
        "next": src.get("next", next_chapter(spec_id) if kind == "Chapter" else None),
        "sections": {},
        "tasks": {},
        "self_assess": {},
    }
    for key, section in (src.get("sections") or {}).items():
        questions = []
        for q in section["questions"]:
            answers = q["answer"] if isinstance(q["answer"], list) and q["type"] == "text" else [q["answer"]]
            hashes = []
            for a in answers:
                normalized = normalize_answer(a, q["type"], q.get("decimals"))
                if not normalized:
                    raise ValueError(f"{spec_id}/{q['id']}: đáp án rỗng")
                hashes.append(hash_answer(spec_id, q["id"], normalized[0]))
            item = {k: q[k] for k in ("id", "type", "topic") if k in q}
            item["answer_hashes"] = hashes
            for k in ("decimals", "lesson"):
                if k in q:
                    item[k] = q[k]
            questions.append(item)
        spec["sections"][key] = {
            "label": section.get("label", DEFAULT_LABELS.get(key, key.title())),
            "threshold": section.get("threshold", 0.8),
            "questions": questions,
        }
    if src.get("coding"):
        spec["tasks"]["coding"] = {
            "label": src["coding"].get("label", DEFAULT_LABELS["coding"]),
            "ids": [t["id"] for t in src["coding"]["tasks"]],
        }
    if src.get("analysis"):
        spec["self_assess"]["analysis"] = {
            "label": src["analysis"].get("label", DEFAULT_LABELS["analysis"]),
            "threshold": src["analysis"].get("threshold", 0.8),
        }
    if src.get("project"):
        spec["tasks"]["project"] = {"label": src["project"].get("label", DEFAULT_LABELS["project"]), "ids": ["project"]}
    return spec


# ----------------------------------------------------------------------------- notebooks
def _options_md(q: dict) -> str:
    opts = q.get("options")
    if not opts:
        return ""
    if isinstance(opts, list):
        opts = {chr(ord("a") + i): o for i, o in enumerate(opts)}
    return "\n".join(f"   - **({k})** {v}" for k, v in opts.items())


def _question_md(q: dict, n: int) -> str:
    extra = ""
    if q["type"] == "numeric":
        extra = f" *(làm tròn {q.get('decimals', 2)} chữ số thập phân)*"
    elif q["type"] == "tf":
        extra = " *(Đúng/Sai)*"
    elif q["type"] == "multi":
        extra = " *(chọn tất cả đáp án đúng)*"
    text = f"{n}. **[{q['id']}]** {q['question'].strip()}{extra}"
    opts = _options_md(q)
    return text + ("\n" + opts if opts else "")


def _answer_literal(q: dict, solution: bool) -> str:
    if not solution:
        return '""' if q["type"] in ("mcq", "multi", "text") else "None"
    a = q["answer"][0] if isinstance(q["answer"], list) and q["type"] == "text" else q["answer"]
    if q["type"] == "numeric":
        return repr(float(a))
    if q["type"] == "tf":
        return repr(bool(a))
    return json.dumps(str(a), ensure_ascii=False)


def build_notebook(src: dict, spec: dict, rel_path: str, solution: bool):
    spec_id, kind = spec["id"], spec["kind"]
    nb = new_notebook()
    cells = []
    banner = "> 📗 **BẢN LỜI GIẢI** — chỉ mở sau khi đã tự làm checkpoint.\n\n" if solution else ""
    thresholds = ", ".join(
        f"{s['label']} ≥ {s['threshold']:.0%}" for s in spec["sections"].values()
    )
    intro = src.get("intro", "").strip()
    head = f"""{colab_badge(rel_path)}

# ✅ {"Bài thi" if kind == "Exam" else "Checkpoint"} — {spec['title']}

{banner}**Cách làm:**
1. Trả lời quiz bằng cách điền vào dict đáp án rồi chạy cell chấm điểm. *Không mở `solutions/` trước khi nộp.*
2. Hoàn thành các task code trong vùng `### START CODE HERE ###` … `### END CODE HERE ###`, chạy cell test.
3. Phần phân tích: viết câu trả lời, rồi **tự chấm trung thực** theo rubric.
4. Chạy cell cuối để xem **STATUS: ✅ PASS / ❌ REVIEW** — kết quả được lưu vào `progress/progress.json`.

**Điều kiện PASS:** {thresholds}{", mọi task code PASS" if spec["tasks"] else ""}{", rubric phân tích ≥ 80%" if spec["self_assess"] else ""}.
"""
    if intro:
        head += "\n" + intro + "\n"
    cells.append(new_markdown_cell(head))
    rel_dir = str(Path(rel_path).parent.as_posix())
    cells.append(new_code_cell(setup_cell_source(rel_dir)))
    record = ", record=False" if solution else ""
    cells.append(new_code_cell(
        "from ml_course.checkpoint import Checkpoint\n"
        "from ml_course.checks import check_close, check_equal, check_greater, check_less, check_shape, check_true\n"
        "import numpy as np\n\n"
        f'cp = Checkpoint("{spec_id}"{record})'
    ))

    part = 0
    for key, section in (src.get("sections") or {}).items():
        part += 1
        label = spec["sections"][key]["label"]
        qs = section["questions"]
        md = f"## Phần {part} — {label}\n\n" + "\n\n".join(_question_md(q, i + 1) for i, q in enumerate(qs))
        cells.append(new_markdown_cell(md))
        lines = [f"{key}_answers = {{"]
        for q in qs:
            lines.append(f'    "{q["id"]}": {_answer_literal(q, solution)},  # {ANSWER_HINT[q["type"]]}')
        lines.append("}")
        lines.append(f'cp.grade("{key}", {key}_answers)')
        cells.append(new_code_cell("\n".join(lines)))
        if solution:
            expl = "\n\n".join(
                f"**[{q['id']}]** Đáp án: `{q['answer']}` — {q.get('explanation', '').strip()}" for q in qs
            )
            cells.append(new_markdown_cell(f"### 💡 Giải thích — {label}\n\n{expl}"))

    if src.get("coding"):
        part += 1
        cells.append(new_markdown_cell(f"## Phần {part} — {spec['tasks']['coding']['label']}"))
        for t in src["coding"]["tasks"]:
            cells.append(new_markdown_cell(f"### Task {t['id']} — {t['title']}\n\n{t['prompt'].strip()}"))
            code = t["code"].rstrip()
            cells.append(new_code_cell(remove_starter_comments(code) if solution else strip_solution_code(code)))
            tests = "\n".join("    " + line if line.strip() else "" for line in t["tests"].rstrip().split("\n"))
            cells.append(new_code_cell(f'with cp.task("{t["id"]}"):\n{tests}'))

    if src.get("analysis"):
        part += 1
        a = src["analysis"]
        rubric = "\n".join(f"- [ ] {item}" for item in a["rubric"])
        cells.append(new_markdown_cell(
            f"## Phần {part} — {spec['self_assess']['analysis']['label']}\n\n{a['prompt'].strip()}\n\n"
            f"**Rubric tự chấm** (đạt khi ≥ {spec['self_assess']['analysis']['threshold']:.0%} số mục):\n{rubric}"
        ))
        answer = a.get("sample_answer", "").strip() if solution else "*(viết câu trả lời của bạn tại đây — double-click để sửa)*"
        cells.append(new_markdown_cell(f"✍️ **{'Bài mẫu' if solution else 'Câu trả lời của bạn'}:**\n\n{answer}"))
        flag = "True" if solution else "False"
        items = "\n".join(f"    {json.dumps(item, ensure_ascii=False)}: {flag}," for item in a["rubric"])
        cells.append(new_code_cell(
            "# Đổi False → True cho mục mà câu trả lời của bạn ĐÃ đáp ứng (trung thực!)\n"
            f"analysis_checklist = {{\n{items}\n}}\n"
            'cp.self_assess("analysis", analysis_checklist)'
        ))

    if src.get("project"):
        part += 1
        p = src["project"]
        cells.append(new_markdown_cell(f"## Phần {part} — {spec['tasks']['project']['label']}\n\n{p['prompt'].strip()}"))
        code = p["code"].rstrip()
        cells.append(new_code_cell(remove_starter_comments(code) if solution else strip_solution_code(code)))
        tests = "\n".join("    " + line if line.strip() else "" for line in p["tests"].rstrip().split("\n"))
        cells.append(new_code_cell(f'with cp.task("project"):\n{tests}'))

    cells.append(new_markdown_cell("## 🏁 Kết quả"))
    cells.append(new_code_cell("cp.report()"))
    nb.cells = cells
    finalize(nb)
    return nb


# ----------------------------------------------------------------------------- main
def build_one(path: Path, kind: str, check: bool) -> int:
    with path.open(encoding="utf-8") as f:
        src = yaml.safe_load(f)
    spec_id = src.get("id", path.stem)
    spec = build_spec(src, spec_id, kind)
    changed = 0
    if kind == "Chapter":
        json_path = ROOT / "quizzes" / "chapter_quizzes" / f"{spec_id}.json"
        learner_rel = f"notebooks/{spec_id}/99_checkpoint.ipynb"
    else:
        json_path = ROOT / "quizzes" / "final_exams" / f"{spec_id}.json"
        learner_rel = f"quizzes/final_exams/{spec_id}.ipynb"
    solution_rel = f"solutions/checkpoints/{spec_id}.ipynb"
    changed += write_if_changed(json_path, json.dumps(spec, ensure_ascii=False, indent=1) + "\n", check)
    changed += write_if_changed(ROOT / learner_rel, notebook_text(build_notebook(src, spec, learner_rel, False)), check)
    changed += write_if_changed(ROOT / solution_rel, notebook_text(build_notebook(src, spec, solution_rel, True)), check)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="*", help="tiền tố tên file YAML")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = 0
    for folder, kind in (("quizzes", "Chapter"), ("exams", "Exam")):
        for path in sorted((ROOT / "solutions" / folder).glob("*.yaml")):
            if args.only and not any(path.stem.startswith(o) for o in args.only):
                continue
            changed += build_one(path, kind, args.check)
    if args.check and changed:
        print(f"{changed} file quiz chưa đồng bộ — chạy: python tools/build_quizzes.py")
        return 1
    print("Quizzes: OK" if not changed else f"Quizzes: {changed} file cập nhật")
    return 0


if __name__ == "__main__":
    sys.exit(main())
