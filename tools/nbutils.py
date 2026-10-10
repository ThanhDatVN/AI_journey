"""Tiện ích dùng chung cho các script trong tools/ (không phụ thuộc package ml_course)."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat

# Windows console mặc định cp1252 → in được tiếng Việt / emoji
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "ThanhDatVN/AI_journey"
BRANCH = "main"

START_RE = re.compile(r"^(\s*)### START CODE HERE ###(.*)$")
END_RE = re.compile(r"^\s*### END CODE HERE ###")
ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=]+)?=(?!=)")
STARTER_RE = re.compile(r"^(\s*)# STARTER:\s?(.*)$")
SETUP_MARKER = "# ⚙️ Setup"


def github_slug() -> str:
    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SLUG
    m = re.search(r'^github_repo\s*=\s*"([^"]+)"', text, flags=re.M)
    return m.group(1) if m else DEFAULT_SLUG


def colab_url(rel_path: str, slug: str | None = None) -> str:
    return f"https://colab.research.google.com/github/{slug or github_slug()}/blob/{BRANCH}/{rel_path}"


def nbviewer_url(rel_path: str, slug: str | None = None) -> str:
    return f"https://nbviewer.org/github/{slug or github_slug()}/blob/{BRANCH}/{rel_path}"


def colab_badge(rel_path: str, slug: str | None = None) -> str:
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_path, slug)})"


def setup_cell_source(rel_dir: str, slug: str | None = None) -> str:
    slug = slug or github_slug()
    return (
        f"{SETUP_MARKER} — chạy được cả local lẫn Google Colab (local: đã `pip install -e .` thì bỏ qua)\n"
        "import sys\n"
        'if "google.colab" in sys.modules:\n'
        f"    !test -d /content/ml-course || git clone -q --depth 1 https://github.com/{slug} /content/ml-course\n"
        "    !pip install -q -e /content/ml-course\n"
        '    sys.path.insert(0, "/content/ml-course/src")\n'
        f"    %cd /content/ml-course/{rel_dir}"
    )


def _assignment_groups(block: list[str], indent: str) -> list[list[str]]:
    """Tên được gán ở mức thụt lề của khối, theo từng câu lệnh: ``a = …`` → [a]; ``a, b = …`` → [a, b].

    Ưu tiên phân tích bằng ``ast`` (bỏ qua dòng nối tiếp của lệnh nhiều dòng; thân ``with`` chạy đúng
    một lần nên tính như cùng mức, vd ``with torch.no_grad(): acc = …``); khối không parse được
    (vd: có magic ``%``/``!``) thì quay về regex từng dòng. Bỏ qua tên ``_`` (giá trị không dùng).
    """
    code = "\n".join(b[len(indent):] if b.startswith(indent) else b.strip() for b in block)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        groups = []
        for b in block:
            if b.startswith(indent) and not b[len(indent):].startswith((" ", "\t")):
                am = ASSIGN_RE.match(b[len(indent):])
                if am:
                    groups.append([am.group(1)])
        return groups
    groups: list[list[str]] = []

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.With):
                visit(node.body)
                continue
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            for t in targets:
                elts = [t] if isinstance(t, ast.Name) else t.elts if isinstance(t, (ast.Tuple, ast.List)) else []
                if elts and all(isinstance(e, ast.Name) for e in elts):
                    names = [e.id for e in elts if e.id != "_"]
                    if names:
                        groups.append(names)

    visit(tree.body)
    return groups


def _is_code(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def strip_solution_code(source: str) -> str:
    """Thay mỗi khối ``### START CODE HERE ###`` … ``### END CODE HERE ###`` bằng chỗ trống cho người học.

    - Dòng ``# STARTER: <code>`` trong khối → giữ lại ``<code>`` (người viết bài chỉ định khung).
    - Nếu không có STARTER: mỗi phép gán ở mức thụt lề của khối → ``tên = None``, gán nhiều tên
      ``a, b = …`` → ``a, b = None, None`` (phong cách AIVN/Coursera).
    - Khung không còn câu lệnh nào (chỉ có comment) → thêm ``pass`` để thân hàm / vòng lặp vẫn hợp lệ.
    """
    out: list[str] = []
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        m = START_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group(1)
        out.append(lines[i])
        block: list[str] = []
        i += 1
        while i < len(lines) and not END_RE.match(lines[i]):
            block.append(lines[i])
            i += 1
        starters = [STARTER_RE.match(b) for b in block]
        stub = [f"{s.group(1)}{s.group(2)}" for s in starters if s]
        if not stub:
            seen: set[str] = set()
            for names in _assignment_groups(block, indent):
                if not set(names) <= seen:
                    seen.update(names)
                    stub.append(f"{indent}{', '.join(names)} = {', '.join(['None'] * len(names))}")
        if not any(_is_code(s) for s in stub):
            stub.append(f"{indent}pass")
        out.extend(stub)
        if i < len(lines):
            out.append(lines[i])  # END marker
        i += 1
    return "\n".join(out)


def remove_starter_comments(source: str) -> str:
    """Trong bản lời giải: bỏ các dòng ``# STARTER:`` để code gọn."""
    return "\n".join(line for line in source.split("\n") if not STARTER_RE.match(line))


def new_notebook():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def finalize(nb) -> None:
    """Id cell cố định + không output → diff sạch khi build lại."""
    for idx, cell in enumerate(nb.cells):
        cell["id"] = f"c{idx:03d}"
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def notebook_text(nb) -> str:
    return nbformat.writes(nb)


def write_if_changed(path: Path, text: str, check: bool) -> bool:
    """Ghi file nếu nội dung khác. Ở chế độ ``check`` chỉ báo khác biệt. Trả về True nếu có thay đổi."""
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        return False
    if check:
        print(f"  ✗ out of date: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"  ✓ wrote {path.relative_to(ROOT)}")
    return True
