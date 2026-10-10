"""Sinh notebook bài tập (cho người học) từ notebook lời giải.

Nguồn sự thật (có lời giải)                     →  Sinh ra (đã xoá lời giải)
    solutions/exercises/<chapter>/<name>.ipynb  →  notebooks/<chapter>/<name>.ipynb
    solutions/projects/<project>/<name>.ipynb   →  projects/<project>/<name>.ipynb
    solutions/practice/<level>/<name>.ipynb     →  exercises/<level>/<name>.ipynb

Quy tắc xoá lời giải:
- cell có tag ``solution`` bị bỏ (vd: đáp án câu hỏi lý thuyết, giải thích);
- code giữa ``### START CODE HERE ###`` và ``### END CODE HERE ###`` được thay bằng khung trống
  (xem ``nbutils.strip_solution_code``);
- output bị xoá.

Dùng:
    python tools/make_exercises.py            # build tất cả
    python tools/make_exercises.py 06         # chỉ các thư mục (chương/project/cấp độ) bắt đầu bằng "06"
    python tools/make_exercises.py --check    # CI: báo lỗi nếu notebook bài tập chưa đồng bộ
"""

from __future__ import annotations

import argparse
import copy
import sys

import nbformat
from nbutils import ROOT, finalize, notebook_text, strip_solution_code, write_if_changed

# (thư mục lời giải, thư mục cho người học) — đường dẫn tương đối so với root repo
MAPPINGS = (("solutions/exercises", "notebooks"), ("solutions/projects", "projects"), ("solutions/practice", "exercises"))


def make_exercise(solution_nb, group: str, source: str = "solutions/exercises", target: str = "notebooks"):
    nb = copy.deepcopy(solution_nb)
    cells = []
    sol_prefix, ex_prefix = f"{source}/{group}", f"{target}/{group}"
    # link từ bản lời giải tới bài học cùng chương: ../../../notebooks/<chương>/bai.ipynb → bai.ipynb
    sibling_link = f"../../../{target}/{group}/"
    for cell in nb.cells:
        if "solution" in cell.get("metadata", {}).get("tags", []):
            continue
        cell.source = cell.source.replace(sol_prefix, ex_prefix).replace(sibling_link, "")
        if cell.cell_type == "code":
            cell.source = strip_solution_code(cell.source)
        cells.append(cell)
    nb.cells = cells
    finalize(nb)
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chapters", nargs="*", help="tiền tố tên chương, vd: 06 07")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = 0
    for source, target in MAPPINGS:
        if not (ROOT / source).exists():
            continue
        for group_dir in sorted(p for p in (ROOT / source).iterdir() if p.is_dir()):
            if args.chapters and not any(group_dir.name.startswith(c) for c in args.chapters):
                continue
            for src in sorted(group_dir.glob("*.ipynb")):
                nb = nbformat.read(src, as_version=4)
                out = make_exercise(nb, group_dir.name, source, target)
                dst = ROOT / target / group_dir.name / src.name
                changed += write_if_changed(dst, notebook_text(out), args.check)
    if args.check and changed:
        print(f"{changed} notebook bài tập chưa đồng bộ — chạy: python tools/make_exercises.py")
        return 1
    print("Exercises: OK" if not changed else f"Exercises: {changed} file cập nhật")
    return 0


if __name__ == "__main__":
    sys.exit(main())
