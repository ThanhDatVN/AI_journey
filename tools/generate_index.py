"""Sinh mục lục tự động (bảng bài học với link GitHub / nbviewer / Colab).

- ``notebooks/<chapter>/README.md``: khối ``<!-- LESSONS:START -->`` … ``<!-- LESSONS:END -->``
- ``SCHEDULE.md``: lịch học theo tuần (phong cách AIVN) từ ``tools/schedule.yaml``
- ``README.md``: khối ``<!-- COURSE:START -->`` … ``<!-- COURSE:END -->`` (bản đồ các chương)

Dùng:
    python tools/generate_index.py           # cập nhật
    python tools/generate_index.py --check   # CI
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import yaml
from nbutils import ROOT, colab_url, nbviewer_url, write_if_changed

sys.path.insert(0, str(ROOT / "src"))
from ml_course.course import CHAPTER_TITLES, PHASES  # noqa: E402

NOTEBOOKS = ROOT / "notebooks"


def notebook_title(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            m = re.match(r"^#{1,2}\s+(.*)", line.strip())
            if m:
                return re.sub(r"\s+", " ", m.group(1)).strip()
    return path.stem


def kind_of(name: str) -> str:
    prefix = name.split("_", 1)[0]
    if prefix == "90":
        return "📝 Bài tập"
    if prefix == "99":
        return "✅ Checkpoint"
    if prefix.startswith("9"):
        return "💻 Assignment"
    return "📖 Bài học"


def links(rel: str) -> str:
    gh = quote(rel)
    return f"[View]({gh}) | [View]({nbviewer_url(gh)}) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(gh)})"


def replace_block(text: str, tag: str, body: str) -> str:
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if start not in text or end not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def chapter_table(chapter_dir: Path) -> str:
    rows = ["| # | Nội dung | Loại | GitHub | nbviewer | Colab |", "|:---:|:---|:---:|:---:|:---:|:---:|"]
    for nb in sorted(chapter_dir.glob("*.ipynb")):
        num = nb.name.split("_", 1)[0]
        rel_from_chapter = nb.name
        gh = quote(rel_from_chapter)
        rel_repo = nb.relative_to(ROOT).as_posix()
        rows.append(
            f"| {num} | {notebook_title(nb)} | {kind_of(nb.name)} | [View]({gh}) | "
            f"[View]({nbviewer_url(quote(rel_repo))}) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(quote(rel_repo))}) |"
        )
    return "\n".join(rows)


def course_map() -> str:
    lines = []
    for i, phase in enumerate(PHASES, start=1):
        lines.append(f"**Phase {i} — {phase.title}**\n")
        lines.append("| Chương | Bài học | Bài tập / Assignment | Checkpoint |")
        lines.append("|:---|:---:|:---:|:---:|")
        for ch in phase.chapters:
            d = NOTEBOOKS / ch
            nbs = sorted(d.glob("*.ipynb")) if d.exists() else []
            n_lessons = sum(not p.name.startswith("9") for p in nbs)
            n_ex = sum(p.name.startswith("9") and not p.name.startswith("99") for p in nbs)
            has_cp = "✅" if (d / "99_checkpoint.ipynb").exists() else "—"
            lines.append(f"| [{ch.split('_')[0]} · {CHAPTER_TITLES[ch]}](notebooks/{ch}/README.md) | {n_lessons} | {n_ex} | {has_cp} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def schedule_kind(item: str) -> tuple[str, str | None]:
    """(biểu tượng, nhãn cột #) cho một mục trong lịch học; nhãn None = bài học được đánh số."""
    name = Path(item).name
    if item.startswith("projects/"):
        return "🛠️", "Project"
    if item.startswith("quizzes/final_exams/"):
        return "🎓", "Thi"
    if item.startswith("exercises/"):
        return "🏋️", "Luyện tập"
    if name.startswith("99"):
        return "✅", "Quiz"
    if name.startswith("90"):
        return "📝", "Bài tập"
    if name.startswith("9"):
        return "💻", "Assignment"
    return "📖", None


def schedule_md() -> str | None:
    src = ROOT / "tools" / "schedule.yaml"
    if not src.exists():
        return None
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    out = [data["header"].rstrip(), ""]
    for week in data["weeks"]:
        out.append(f"### Tuần {week['week']} — {week['title']}\n")
        if week.get("goal"):
            out.append(f"🎯 {week['goal']}\n")
        out.append("| # | Nội dung | GitHub | nbviewer | Colab |")
        out.append("|:---:|:---|:---:|:---:|:---:|")
        lesson_no = 0
        for item in week["items"]:
            path = ROOT / item
            label = (notebook_title(path) if path.exists() else item).removeprefix("✅ ")
            prefix, num = schedule_kind(item)
            if num is None:
                lesson_no += 1
                num = lesson_no
            out.append(f"| {num} | {prefix} {label} <br/><sub>`{item}`</sub> | {links(item)} |")
        out.append("")
    if data.get("footer"):
        out.append(data["footer"].rstrip())
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = 0
    for chapter_dir in sorted(p for p in NOTEBOOKS.iterdir() if p.is_dir()):
        readme = chapter_dir / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            changed += write_if_changed(readme, replace_block(text, "LESSONS", chapter_table(chapter_dir)), args.check)
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        changed += write_if_changed(readme, replace_block(text, "COURSE", course_map()), args.check)
    sched = schedule_md()
    if sched is not None:
        changed += write_if_changed(ROOT / "SCHEDULE.md", sched, args.check)
    if args.check and changed:
        print(f"{changed} file mục lục chưa đồng bộ — chạy: python tools/generate_index.py")
        return 1
    print("Index: OK" if not changed else f"Index: {changed} file cập nhật")
    return 0


if __name__ == "__main__":
    sys.exit(main())
