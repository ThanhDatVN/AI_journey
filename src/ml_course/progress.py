"""Lưu và hiển thị tiến độ học.

Kết quả mỗi checkpoint được ghi vào ``progress/progress.json`` (ở root repo).

CLI::

    python -m ml_course.progress                 # in bảng tiến độ
    python -m ml_course.progress --update-readme # cập nhật khối PROGRESS trong README.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .course import CHAPTERS, PHASES, chapter_label

README_START = "<!-- PROGRESS:START -->"
README_END = "<!-- PROGRESS:END -->"


def repo_root() -> Path:
    """Tìm thư mục gốc của repo khoá học.

    Thứ tự: biến môi trường ``ML_COURSE_ROOT`` → đi ngược từ thư mục hiện tại tìm
    ``pyproject.toml`` + ``quizzes/`` → vị trí của package (cài editable).
    """
    env = os.environ.get("ML_COURSE_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "quizzes").is_dir():
            return candidate
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "quizzes").is_dir():
        return package_root
    return cwd


def progress_file() -> Path:
    return repo_root() / "progress" / "progress.json"


def load_progress() -> dict[str, Any]:
    path = progress_file()
    if not path.exists():
        return {"chapters": {}}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def record_result(chapter: str, payload: dict[str, Any]) -> Path:
    """Ghi kết quả checkpoint (giữ lại lần làm tốt nhất + lịch sử các lần làm)."""
    data = load_progress()
    chapters = data.setdefault("chapters", {})
    entry = chapters.setdefault(chapter, {"attempts": []})
    payload = {**payload, "timestamp": datetime.now().isoformat(timespec="seconds")}
    entry["attempts"].append(payload)
    entry["attempts"] = entry["attempts"][-20:]
    entry["latest"] = payload
    if payload.get("status") == "PASS":
        entry["passed"] = True
        entry.setdefault("first_passed_at", payload["timestamp"])
    path = progress_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def passed_chapters(data: dict[str, Any] | None = None) -> set[str]:
    data = data if data is not None else load_progress()
    return {ch for ch, entry in data.get("chapters", {}).items() if entry.get("passed")}


def bar(fraction: float, width: int = 10) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def render_progress(data: dict[str, Any] | None = None) -> str:
    done = passed_chapters(data)
    lines = []
    name_width = max(len(p.title) for p in PHASES)
    for phase in PHASES:
        n_done = sum(ch in done for ch in phase.chapters)
        frac = n_done / len(phase.chapters)
        lines.append(
            f"{phase.title:<{name_width}}  {bar(frac)} {frac:>4.0%}  ({n_done}/{len(phase.chapters)})"
        )
    total = sum(ch in done for ch in CHAPTERS)
    lines.append("")
    lines.append(f"{'TOTAL':<{name_width}}  {bar(total / len(CHAPTERS))} {total / len(CHAPTERS):>4.0%}  ({total}/{len(CHAPTERS)})")
    nxt = next((ch for ch in CHAPTERS if ch not in done), None)
    lines.append(f"NEXT: {chapter_label(nxt)}" if nxt else "🎓 Hoàn thành toàn bộ khoá học!")
    return "\n".join(lines)


def update_readme(readme: Path | None = None) -> bool:
    readme = readme or repo_root() / "README.md"
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    block = f"{README_START}\n```text\n{render_progress()}\n```\n{README_END}"
    readme.write_text(head + block + tail, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tiến độ học Machine Learning: Zero → Advanced")
    parser.add_argument("--update-readme", action="store_true", help="ghi bảng tiến độ vào README.md")
    args = parser.parse_args(argv)
    print(render_progress())
    if args.update_readme:
        ok = update_readme()
        print("\nREADME.md đã được cập nhật." if ok else "\nKhông tìm thấy khối PROGRESS trong README.md.")


if __name__ == "__main__":
    main()
