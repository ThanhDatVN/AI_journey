"""Chạy (execute) notebook để bảo đảm khoá học luôn chạy được — dùng local và trong CI.

Mặc định BỎ QUA các notebook người học cần tự điền: ``notebooks/**/9*_*.ipynb``
(bài tập 90_…, assignment 91_…, checkpoint 99_…). Bản lời giải của chúng trong ``solutions/``
thì được chạy; riêng ``solutions/checkpoints`` còn phải in ra ``STATUS: ✅ PASS``.

Dùng:
    python tools/run_notebooks.py notebooks/06_linear_regression
    python tools/run_notebooks.py notebooks solutions --workers 4
    python tools/run_notebooks.py path/to/one.ipynb --timeout 900
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect(targets: list[str], include_exercises: bool) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        for match in glob(target) or [target]:
            p = Path(match)
            if p.is_dir():
                paths.extend(sorted(p.rglob("*.ipynb")))
            elif p.suffix == ".ipynb":
                paths.append(p)
    out = []
    for p in paths:
        if ".ipynb_checkpoints" in p.parts:
            continue
        rel = p.resolve().relative_to(ROOT).as_posix()
        if not include_exercises and rel.startswith("notebooks/") and p.name.startswith("9"):
            continue
        if not include_exercises and rel.startswith("quizzes/"):
            continue
        out.append(p.resolve())
    return sorted(set(out))


def run_one(path: str, timeout: int) -> tuple[str, bool, float, str]:
    import nbformat
    from nbclient import NotebookClient

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ["ML_COURSE_NO_RECORD"] = "1"
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    p = Path(path)
    start = time.perf_counter()
    nb = nbformat.read(p, as_version=4)
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(p.parent)}}).execute()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        return path, False, time.perf_counter() - start, msg[-3000:]
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith("solutions/checkpoints/"):
        text = "".join(
            o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
        )
        if "STATUS: ✅ PASS" not in text:
            tail = text[-1500:]
            return path, False, time.perf_counter() - start, f"checkpoint solution không PASS:\n{tail}"
    return path, True, time.perf_counter() - start, ""


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=["notebooks", "solutions"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--include-exercises", action="store_true")
    args = parser.parse_args(argv)
    paths = collect(args.targets, args.include_exercises)
    print(f"Executing {len(paths)} notebooks with {args.workers} workers…")
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, str(p), args.timeout) for p in paths]
        for fut in as_completed(futures):
            path, ok, secs, msg = fut.result()
            rel = Path(path).relative_to(ROOT).as_posix()
            print(f"{'✅' if ok else '❌'} {secs:6.1f}s  {rel}", flush=True)
            if not ok:
                failures.append((rel, msg))
    if failures:
        print(f"\n{len(failures)} notebook lỗi:")
        for rel, msg in failures:
            print(f"\n──── {rel}\n{msg}")
        return 1
    print("\nTất cả notebook chạy thành công.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
