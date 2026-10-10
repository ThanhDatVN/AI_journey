"""Checkpoint engine — chấm quiz và quyết định PASS / REVIEW cho từng chương.

Một checkpoint gồm các thành phần (tuỳ chương có thể thiếu vài phần):

=================  ===========================================  =====================
Thành phần         Cách chấm                                     Điều kiện đạt
=================  ===========================================  =====================
Theory Quiz        trắc nghiệm / đúng-sai, đáp án đã được hash   ≥ ngưỡng (mặc định 80%)
Math Quiz          tính tay, đáp án số (làm tròn theo đề)        ≥ ngưỡng
Coding Exercise    ``with cp.task("c01"):`` chạy test             tất cả task PASS
Model Analysis     tự đánh giá theo rubric (trung thực!)          ≥ ngưỡng rubric
Mini Project       ``with cp.task("project"):`` metric ≥ mốc      PASS
=================  ===========================================  =====================

Đáp án quiz được lưu dưới dạng hash trong ``quizzes/``; lời giải + giải thích nằm trong
``solutions/`` — chỉ mở sau khi đã nộp bài.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .course import chapter_label
from .progress import record_result, repo_root

TRUE_WORDS = {"true", "t", "đúng", "dung", "d", "1", "yes", "y", "có"}
FALSE_WORDS = {"false", "f", "sai", "s", "0", "no", "n", "không"}


# ----------------------------------------------------------------------------- hashing
def normalize_answer(answer: Any, qtype: str, decimals: int | None = None) -> list[str]:
    """Chuẩn hoá câu trả lời → danh sách chuỗi ứng viên (numeric có dung sai ±1 đơn vị cuối)."""
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return []
    if qtype == "numeric":
        d = 2 if decimals is None else int(decimals)
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return []
        if math.isnan(value):
            return []
        step = 10.0**-d
        candidates = []
        for delta in (0.0, -step, step):
            v = round(value + delta, d)
            v = 0.0 if v == 0 else v  # tránh "-0.00"
            candidates.append(f"{v:.{d}f}")
        return candidates
    if qtype == "tf":
        if isinstance(answer, bool):
            return ["true" if answer else "false"]
        word = str(answer).strip().lower()
        if word in TRUE_WORDS:
            return ["true"]
        if word in FALSE_WORDS:
            return ["false"]
        return [word]
    if qtype == "multi":
        if isinstance(answer, str):
            letters = [ch for ch in answer.lower() if ch.isalpha()]
        else:
            letters = [str(a).strip().lower() for a in answer]
        return ["".join(sorted(set(letters)))]
    if qtype == "mcq":
        text = str(answer).strip().lower().strip("().: ")
        return [text[:1] if text else ""]
    # text: so khớp không phân biệt hoa thường / khoảng trắng
    return [" ".join(str(answer).strip().lower().split())]


def hash_answer(salt: str, qid: str, normalized: str) -> str:
    return hashlib.sha256(f"{salt}|{qid}|{normalized}".encode()).hexdigest()[:20]


def is_correct(spec_id: str, question: Mapping[str, Any], answer: Any) -> bool:
    candidates = normalize_answer(answer, question["type"], question.get("decimals"))
    hashes = set(question["answer_hashes"])
    return any(hash_answer(spec_id, question["id"], c) in hashes for c in candidates)


# ----------------------------------------------------------------------------- results
@dataclass
class ComponentResult:
    label: str
    score: float = 0.0
    total: float = 0.0
    passed: bool = False
    attempted: bool = False
    detail: str = ""
    weak_topics: list[tuple[str, str]] = field(default_factory=list)  # (topic, lesson)

    def _ipython_display_(self) -> None:
        """Kết quả đã được in khi chấm — không hiển thị thêm repr trong notebook."""

    def summary(self) -> str:
        if not self.attempted:
            return "NOT DONE"
        if self.detail:
            return self.detail
        return "PASS" if self.passed else "FAIL"


def load_spec(spec_id: str) -> dict[str, Any]:
    root = repo_root()
    for folder in ("quizzes/chapter_quizzes", "quizzes/final_exams"):
        path = root / folder / f"{spec_id}.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Không tìm thấy quiz '{spec_id}'. Hãy chạy notebook từ trong repo, hoặc đặt biến "
        f"môi trường ML_COURSE_ROOT trỏ tới thư mục gốc của repo (hiện tại: {root})."
    )


class Checkpoint:
    """Engine chấm điểm của một chương (hoặc một bài thi tổng hợp)."""

    def __init__(self, spec_id: str, record: bool = True):
        self.spec_id = spec_id
        self.spec = load_spec(spec_id)
        self.record = record and os.environ.get("ML_COURSE_NO_RECORD") != "1"
        self.results: dict[str, ComponentResult] = {}
        self._task_results: dict[str, bool] = {}
        for key, section in self.spec.get("sections", {}).items():
            self.results[key] = ComponentResult(label=section["label"])
        for key, tasks in self.spec.get("tasks", {}).items():
            self.results[key] = ComponentResult(label=tasks["label"])
        for key, rubric in self.spec.get("self_assess", {}).items():
            self.results[key] = ComponentResult(label=rubric["label"])
        title = self.spec.get("title", spec_id)
        print(f"📋 Checkpoint: {title}")
        print("   Thành phần: " + ", ".join(r.label for r in self.results.values()))

    # ------------------------------------------------------------------ quiz sections
    def grade(self, section: str, answers: Mapping[str, Any]) -> ComponentResult:
        """Chấm một phần quiz. ``answers`` là dict {question_id: câu trả lời}."""
        spec = self.spec["sections"][section]
        result = self.results[section]
        questions = spec["questions"]
        correct, weak, marks = 0, [], []
        for q in questions:
            ans = answers.get(q["id"])
            if not normalize_answer(ans, q["type"], q.get("decimals")):
                marks.append(f"{q['id']} ⚠️ (chưa trả lời)")
                weak.append((q["topic"], q.get("lesson", "")))
            elif is_correct(self.spec_id, q, ans):
                correct += 1
                marks.append(f"{q['id']} ✅")
            else:
                marks.append(f"{q['id']} ❌ [{q['topic']}]")
                weak.append((q["topic"], q.get("lesson", "")))
        total = len(questions)
        threshold = spec.get("threshold", 0.8)
        frac = correct / total if total else 0.0
        result.score, result.total, result.attempted = correct, total, True
        result.passed = frac >= threshold - 1e-9
        result.detail = f"{correct}/{total}"
        result.weak_topics = _dedupe(weak)
        icon = "✅" if result.passed else "❌"
        print(f"{spec['label']}: {correct}/{total} ({frac:.0%}) {icon}  — ngưỡng đạt {threshold:.0%}")
        for i in range(0, len(marks), 4):
            print("   " + "   ".join(marks[i : i + 4]))
        return result

    # ------------------------------------------------------------------ coding tasks
    @contextmanager
    def task(self, task_id: str) -> Iterator[None]:
        """Chạy test cho một task; mọi lỗi được bắt lại và ghi FAIL."""
        component = self._component_of(task_id)
        print(f"── Task {task_id} ──")
        ok = False
        try:
            yield
            ok = True
        except NotImplementedError:
            print("⏳ Chưa làm (NotImplementedError).")
        except AssertionError as exc:
            print(str(exc) or "❌ Test thất bại.")
        except Exception as exc:  # noqa: BLE001 — muốn báo mọi lỗi cho người học
            print(f"💥 Lỗi khi chạy: {type(exc).__name__}: {exc}")
            print("".join(traceback.format_exception_only(type(exc), exc)).strip())
        self._task_results[task_id] = ok
        print(f"Task {task_id}: {'PASS ✅' if ok else 'FAIL ❌'}")
        self._refresh_tasks(component)

    def _component_of(self, task_id: str) -> str:
        for key, tasks in self.spec.get("tasks", {}).items():
            if task_id in tasks["ids"]:
                return key
        raise KeyError(f"Task '{task_id}' không có trong checkpoint này.")

    def _refresh_tasks(self, component: str) -> None:
        tasks = self.spec["tasks"][component]
        result = self.results[component]
        ids = tasks["ids"]
        done = [t for t in ids if t in self._task_results]
        passed = [t for t in ids if self._task_results.get(t)]
        result.attempted = len(done) == len(ids)
        result.score, result.total = len(passed), len(ids)
        result.passed = len(passed) == len(ids)
        status = "PASS" if result.passed else ("FAIL" if result.attempted else "IN PROGRESS")
        result.detail = f"{status} ({len(passed)}/{len(ids)} tasks)" if len(ids) > 1 else status
        result.weak_topics = [(f"task {t}", "") for t in ids if t in done and not self._task_results[t]]

    # ------------------------------------------------------------------ self assessment
    def self_assess(self, component: str, checklist: Mapping[str, bool]) -> ComponentResult:
        """Tự chấm theo rubric. Hãy trung thực — mục tiêu là hiểu, không phải điểm số."""
        spec = self.spec["self_assess"][component]
        result = self.results[component]
        items = list(checklist.items())
        n_true = sum(bool(v) for _, v in items)
        total = max(len(items), 1)
        threshold = spec.get("threshold", 0.8)
        result.score, result.total, result.attempted = n_true, total, True
        result.passed = n_true / total >= threshold - 1e-9
        result.detail = "PASS" if result.passed else f"FAIL ({n_true}/{total})"
        result.weak_topics = [(item, "") for item, v in items if not v]
        for item, v in items:
            print(f"   {'☑' if v else '☐'} {item}")
        print(f"{spec['label']}: {n_true}/{total} → {'PASS ✅' if result.passed else 'FAIL ❌'}")
        return result

    # ------------------------------------------------------------------ report
    def status(self) -> str:
        if not all(r.attempted for r in self.results.values()):
            return "INCOMPLETE"
        return "PASS" if all(r.passed for r in self.results.values()) else "REVIEW"

    def report(self) -> None:
        """In bảng kết quả + STATUS và lưu vào progress/progress.json. Xem ``status()`` để lấy trạng thái."""
        status = self.status()
        title = self.spec.get("title", self.spec_id)
        width = max(len(r.label) for r in self.results.values()) + 2
        lines = ["═" * 46, f"{self.spec.get('kind', 'Chapter')}: {title}", "─" * 46]
        for r in self.results.values():
            lines.append(f"{r.label + ':':<{width}} {r.summary()}")
        lines.append("─" * 46)
        icon = {"PASS": "✅", "REVIEW": "❌", "INCOMPLETE": "⏳"}[status]
        lines.append(f"STATUS: {icon} {status}")
        if status == "PASS":
            nxt = self.spec.get("next")
            lines.append(f"NEXT: {chapter_label(nxt)}" if nxt else "NEXT: 🎉 hoàn thành!")
        else:
            missing = [r.label for r in self.results.values() if not r.attempted]
            if missing:
                lines.append("Chưa làm: " + ", ".join(missing))
            weak = _dedupe([w for r in self.results.values() for w in r.weak_topics])
            if weak:
                lines.append("")
                lines.append("Weak topics:")
                for topic, lesson in weak:
                    lines.append(f"  - {topic}" + (f"   → ôn lại {lesson}" if lesson else ""))
        lines.append("═" * 46)
        text = "\n".join(lines)
        print(text)
        if self.record:
            payload = {
                "status": status,
                "components": {
                    k: {"score": r.score, "total": r.total, "passed": r.passed, "attempted": r.attempted}
                    for k, r in self.results.items()
                },
            }
            path = record_result(self.spec_id, payload)
            print(f"\n💾 Đã lưu kết quả vào {path.relative_to(repo_root()) if path.is_relative_to(repo_root()) else path}")


def _dedupe(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for item in items:
        if item[0] not in seen:
            seen.add(item[0])
            out.append(item)
    return out


__all__ = ["Checkpoint", "hash_answer", "normalize_answer", "is_correct", "load_spec"]
