"""Đổi slug GitHub (owner/repo) trong mọi link Colab / nbviewer / git clone của khoá học.

Sau khi fork hoặc đẩy repo lên GitHub của bạn:
    python tools/set_github_repo.py <owner>/<repo>
"""

from __future__ import annotations

import re
import sys

from nbutils import ROOT, github_slug

SKIP_DIRS = {".git", ".venv", "AIVN-Machine-Learning-master", "node_modules"}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not re.fullmatch(r"[\w.-]+/[\w.-]+", argv[0]):
        print(__doc__)
        return 1
    new, old = argv[0], github_slug()
    if new == old:
        print(f"Slug đã là {new}.")
        return 0
    n_files = 0
    for path in ROOT.rglob("*"):
        if path.suffix not in {".ipynb", ".md", ".toml", ".yml", ".yaml", ".py"} or not path.is_file():
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
            n_files += 1
    print(f"Đã đổi {old} → {new} trong {n_files} file.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
