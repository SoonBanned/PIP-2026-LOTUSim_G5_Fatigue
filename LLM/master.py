from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from database import Database


def _import_llm_class():
    try:
        from llm import LLM  # type: ignore

        return LLM
    except Exception:
        from .llm import LLM  # type: ignore

        return LLM


def _import_interface_class():
    try:
        from interface import Interface  # type: ignore

        return Interface
    except Exception:
        from .interface import Interface  # type: ignore

        return Interface


@dataclass
class MasterConfig:
    host: str = "127.0.0.1"
    port: int = 5001
    results_dir: str | Path = Path(__file__).resolve().parent / "results"
    llm_model: str = "llama3.1_fine"
    db_path: str | Path = Path(__file__).resolve().parent / "users_db.json"
    grading_queue_size: int = 32


class Master:
    """Coordinator between the UI (Interface) and the LLM layer.

    This intentionally defines callback methods used by Interface, and keeps
    placeholders for future Master -> Interface calls.
    """

    def __init__(
        self,
        config: MasterConfig | None = None,
        *,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or MasterConfig()
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.llm_error: str | None = None
        self.llm: Any | None = None

        # Keep init non-blocking: we load the LLM in the background once the
        # asyncio loop is running, so the web UI can start serving immediately.
        self._llm_kwargs = dict(llm_kwargs or {})
        self._llm_init_lock = asyncio.Lock()
        self._llm_init_task: asyncio.Task[None] | None = None

        self.db = Database(str(self.config.db_path))

        self._tests_lock = asyncio.Lock()
        self._tests: dict[str, dict[str, Any]] = {}

        # Queue contains grading jobs: one question per job.
        self._grading_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=int(self.config.grading_queue_size)
        )
        self._grading_worker_task: asyncio.Task[None] | None = None

        print("Starting the Interface")
        Interface = _import_interface_class()
        self.interface = Interface(
            master=self, host=self.config.host, port=self.config.port
        )

        # LLM init is started later via start_llm_background() / ensure_llm_ready().

        # Quiz is now per-test session (stored in self._tests[test_id]["quiz"]).

    def start_llm_background(self) -> None:
        """Kick off LLM initialization without blocking server startup."""
        if self._llm_init_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet; caller should invoke from an async context.
            return
        self._llm_init_task = loop.create_task(self.ensure_llm_ready())

    def start_grading_worker_background(self) -> None:
        if self._grading_worker_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._grading_worker_task = loop.create_task(self._grading_worker())

    async def ensure_llm_ready(self) -> None:
        """Ensure self.llm is initialized; safe to call multiple times."""
        if self.llm is not None:
            return

        async with self._llm_init_lock:
            if self.llm is not None:
                return

            print("Starting the LLM (background)")
            LLM = _import_llm_class()

            def _init_sync() -> Any:
                return LLM(model=self.config.llm_model, **self._llm_kwargs)

            try:
                self.llm = await asyncio.to_thread(_init_sync)
                self.llm_error = None
                print("LLM initialized")
            except Exception as e:
                self.llm = None
                self.llm_error = f"LLM init failed: {e}"
                print(self.llm_error)

    # -------------------------
    # Users / Tests
    # -------------------------
    @staticmethod
    def _sanitize_name(name: str) -> str:
        return " ".join(str(name or "").strip().split())

    async def register_user(self, *, name: str) -> Dict[str, Any]:
        nm = self._sanitize_name(name)
        if not nm:
            return {"ok": False, "error": "Name is required"}
        return {"ok": True, "name": nm}

    async def create_test(
        self, *, name: str, n: int = 4, difficulty: str = "easy"
    ) -> Dict[str, Any]:
        nm = self._sanitize_name(name)
        if not nm:
            return {"ok": False, "error": "Name is required"}

        test_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat(timespec="seconds")
        session = {
            "test_id": test_id,
            "name": nm,
            "date": created_at,
            "quiz": [],
            "answers": [],
            "totalTestTime": None,
            "submitted": False,
            "grading": {
                "status": "not_submitted",
                "progress": 0,
                "total": 0,
                "per_question": [],
                "average_score": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
            },
        }

        async with self._tests_lock:
            self._tests[test_id] = session

        # If LLM is still loading, wait here (UI shows the big overlay spinner).
        await self.ensure_llm_ready()

        quiz_items = await self._generate_quiz(n=int(n), difficulty=str(difficulty))
        async with self._tests_lock:
            s = self._tests.get(test_id)
            if s:
                s["quiz"] = quiz_items
                total = len(quiz_items)
                s["grading"]["total"] = total
                s["answers"] = [None] * total
                s["grading"]["per_question"] = [
                    {
                        "index": i,
                        "status": "not_answered",
                        "score_total": None,
                        "feedback": None,
                    }
                    for i in range(total)
                ]

        return {
            "ok": True,
            "test_id": test_id,
            "total": len(quiz_items),
            "difficulty": str(difficulty),
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }

    @staticmethod
    def _ensure_list_length(lst: list[Any], length: int, fill: Any = None) -> None:
        if length <= 0:
            return
        if len(lst) < length:
            lst.extend([fill] * (length - len(lst)))

    async def submit_answer(
        self, *, test_id: str, index: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store a single answer and enqueue grading for that question.

        This is used to grade incrementally after each question, so the user
        doesn't wait for cumulative grading at the end.
        """

        tid = str(test_id)
        i = int(index)

        # Defensive: ensure background tasks exist even if caller didn't start them.
        self.start_llm_background()
        self.start_grading_worker_background()

        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}

        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return {"ok": False, "error": "Unknown test_id"}

            quiz = session.get("quiz") or []
            total = len(quiz) if isinstance(quiz, list) else 0
            if i < 0 or i >= total:
                return {"ok": False, "error": "Invalid question index"}

            answers = session.get("answers")
            if not isinstance(answers, list):
                answers = []
            self._ensure_list_length(answers, total, None)

            # Persist the answer (UI sends {answer, timeSpent, keystrokes, question?}).
            ans_obj = {
                "question": payload.get("question"),
                "answer": payload.get("answer"),
                "timeSpent": payload.get("timeSpent"),
                "keystrokes": payload.get("keystrokes"),
            }
            answers[i] = ans_obj
            session["answers"] = answers

            per_q = session.get("grading", {}).get("per_question")
            if not isinstance(per_q, list):
                per_q = []
            self._ensure_list_length(per_q, total, None)
            if not isinstance(per_q[i], dict):
                per_q[i] = {"index": i}

            status = str(per_q[i].get("status") or "not_answered")
            # Avoid flooding the queue if already queued/processing.
            if status not in {"queued", "processing"}:
                per_q[i]["status"] = "queued"
                per_q[i]["score_total"] = None
                per_q[i]["feedback"] = None
            session["grading"]["per_question"] = per_q

            # Overall grading status reflects background work.
            if session["grading"].get("status") in {"not_submitted", "done", "failed"}:
                session["grading"]["status"] = "processing"
                if not session["grading"].get("started_at"):
                    session["grading"]["started_at"] = datetime.now().isoformat(
                        timespec="seconds"
                    )

        # Enqueue grading job.
        try:
            self._grading_queue.put_nowait({"test_id": tid, "index": i})
            print(f"[grading] queued test_id={tid} index={i}")
        except asyncio.QueueFull:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session and isinstance(session.get("grading"), dict):
                    per_q = session["grading"].get("per_question")
                    if (
                        isinstance(per_q, list)
                        and i < len(per_q)
                        and isinstance(per_q[i], dict)
                    ):
                        per_q[i]["status"] = "failed"
                        per_q[i]["feedback"] = "Grading queue is full"
                    session["grading"]["error"] = "Grading queue is full"
                    session["grading"]["status"] = "failed"
            return {"ok": False, "error": "Grading queue is full"}

        return {"ok": True, "test_id": tid, "index": i, "status": "queued"}

    async def get_question_for_test(
        self, *, test_id: str, index: int
    ) -> Dict[str, Any]:
        tid = str(test_id)
        i = int(index)
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}

        quiz = session.get("quiz") or []
        if not isinstance(quiz, list):
            quiz = []
        total = len(quiz)
        if i < 0 or i >= total:
            return {"ok": True, "done": True, "index": i, "total": total}

        item = quiz[i]
        question = (item.get("question") if isinstance(item, dict) else None) or str(
            item
        )
        image = item.get("image") if isinstance(item, dict) else None
        return {
            "ok": True,
            "done": False,
            "test_id": tid,
            "index": i,
            "total": total,
            "question": question,
            "image": image,
        }

    async def submit_answers(
        self, *, test_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        tid = str(test_id)

        # Defensive: ensure background tasks exist even if caller didn't start them.
        self.start_llm_background()
        self.start_grading_worker_background()
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}

        answers = payload.get("results") or payload.get("answers") or []
        total_time = payload.get("totalTestTime")

        # Keep endpoint for backwards-compatibility and finalization.
        # If the UI already submitted per-question answers, this mainly sets
        # totalTestTime and ensures any missing grading jobs are queued.

        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return {"ok": False, "error": "Unknown test_id"}

            session["totalTestTime"] = total_time
            session["submitted"] = True
            session["grading"]["error"] = None

            quiz = session.get("quiz") or []
            total = len(quiz) if isinstance(quiz, list) else 0

            if isinstance(answers, list) and answers:
                # If the UI sends all answers at once, store them as-is.
                session["answers"] = answers

            if session["grading"].get("status") == "not_submitted":
                session["grading"]["status"] = "processing"
                session["grading"]["started_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )

            per_q = session["grading"].get("per_question")
            if not isinstance(per_q, list):
                per_q = []
            self._ensure_list_length(per_q, total, None)
            session["grading"]["per_question"] = per_q

            # If grading already finished before final submit, finalize status now.
            all_done = True
            for item in per_q:
                if not isinstance(item, dict) or str(item.get("status")) not in {
                    "done",
                    "failed",
                }:
                    all_done = False
                    break
            if all_done and total:
                session["grading"]["status"] = "done"
                session["grading"]["finished_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )

        # Enqueue grading for any question that has an answer and isn't done.
        queued_any = False
        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return {"ok": False, "error": "Unknown test_id"}
            quiz = session.get("quiz") or []
            total = len(quiz) if isinstance(quiz, list) else 0
            answers_list = session.get("answers")
            per_q = session.get("grading", {}).get("per_question")

        if isinstance(answers_list, list) and isinstance(per_q, list):
            for i in range(min(len(answers_list), len(per_q), total)):
                has_answer = (
                    isinstance(answers_list[i], dict)
                    and str(answers_list[i].get("answer") or "").strip()
                )
                if not has_answer:
                    continue
                st = str(per_q[i].get("status") or "")
                if st in {"done", "processing", "queued"}:
                    continue
                try:
                    self._grading_queue.put_nowait({"test_id": tid, "index": i})
                    queued_any = True
                    async with self._tests_lock:
                        session = self._tests.get(tid)
                        if session and isinstance(session.get("grading"), dict):
                            pq = session["grading"].get("per_question")
                            if (
                                isinstance(pq, list)
                                and i < len(pq)
                                and isinstance(pq[i], dict)
                            ):
                                pq[i]["status"] = "queued"
                except asyncio.QueueFull:
                    async with self._tests_lock:
                        session = self._tests.get(tid)
                        if session:
                            session["grading"]["status"] = "failed"
                            session["grading"]["error"] = "Grading queue is full"
                    return {"ok": False, "error": "Grading queue is full"}

        return {
            "ok": True,
            "test_id": tid,
            "status": "processing",
            "queued": queued_any,
        }

    async def get_grading_status(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}
        grading = session.get("grading") or {}
        per_q = grading.get("per_question") if isinstance(grading, dict) else None
        progress = grading.get("progress", 0)
        total = grading.get("total", 0)
        if isinstance(per_q, list) and total:
            done_count = 0
            for item in per_q:
                if isinstance(item, dict) and str(item.get("status")) == "done":
                    done_count += 1
            progress = done_count
        return {
            "ok": True,
            "test_id": tid,
            "status": grading.get("status"),
            "progress": progress,
            "total": total,
            "error": grading.get("error"),
        }

    async def get_grading_result(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}

        grading = session.get("grading") or {}
        return {
            "ok": True,
            "test_id": tid,
            "status": grading.get("status"),
            "grading": grading,
            "answers": session.get("answers") or [],
            "quiz": [
                {"question": q.get("question"), "truth_answer": q.get("truth_answer")}
                for q in (session.get("quiz") or [])
                if isinstance(q, dict)
            ],
        }

    # -------------------------
    # Quiz + grading internals
    # -------------------------
    def _fallback_quiz(self, n: int) -> list[dict[str, Any]]:
        # Minimal fallback if the LLM isn't available.
        base = [
            "Décris ton niveau de vigilance actuel.",
            "As-tu eu des micro-somnolences récemment ?",
            "Depuis combien de temps es-tu réveillé(e) aujourd’hui ?",
            "As-tu des difficultés de concentration en ce moment ?",
        ]
        out: list[dict[str, Any]] = []
        for i in range(max(1, int(n))):
            out.append({"question": base[i % len(base)], "difficulty": "easy"})
        return out

    async def _generate_quiz(
        self, *, n: int = 4, difficulty: str = "easy"
    ) -> list[dict[str, Any]]:
        n = max(1, int(n))
        difficulty = str(difficulty or "easy")

        if self.llm is None:
            return self._fallback_quiz(n)

        difficulty_lists = [difficulty] * n
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                items = await asyncio.to_thread(
                    self.llm.generate_quizz, n, difficulty_lists
                )
                if not isinstance(items, list) or not items:
                    raise RuntimeError("LLM returned empty quiz")
                return items
            except Exception as e:
                last_err = e
                self.llm_error = f"Quiz generation failed: {e}"
                print(self.llm_error)
                if attempt < 3:
                    print("Retrying quiz generation")
                    await asyncio.sleep(0.2)
        if last_err is not None:
            self.llm_error = f"Quiz generation failed (fallback used): {last_err}"
        return self._fallback_quiz(n)

    async def _grading_worker(self) -> None:
        while True:
            job = await self._grading_queue.get()
            try:
                await self._process_grading_job(job=job)
            finally:
                self._grading_queue.task_done()

    async def _process_grading_job(self, *, job: dict[str, Any]) -> None:
        tid = str(job.get("test_id") or "")
        index = int(job.get("index", -1))
        print(f"[grading] start  test_id={tid} index={index}")
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return

        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return
            grading = session.get("grading")
            if not isinstance(grading, dict):
                grading = {"status": "processing"}
                session["grading"] = grading

            grading["status"] = "processing"
            grading.setdefault(
                "started_at", datetime.now().isoformat(timespec="seconds")
            )
            grading["error"] = None

        await self.ensure_llm_ready()
        if self.llm is None:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session:
                    session["grading"]["status"] = "failed"
                    session["grading"]["error"] = (
                        session["grading"].get("error")
                        or self.llm_error
                        or "LLM not available"
                    )
            return

        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return
            quiz = session.get("quiz") or []
            answers = session.get("answers") or []
            grading = session.get("grading") or {}
            per_q = grading.get("per_question") if isinstance(grading, dict) else None
        if not isinstance(quiz, list):
            quiz = []
        if not isinstance(answers, list):
            answers = []

        total = len(quiz)
        if index < 0 or index >= total:
            return

        if not (
            isinstance(per_q, list)
            and index < len(per_q)
            and isinstance(per_q[index], dict)
        ):
            # Ensure per_question exists.
            async with self._tests_lock:
                session = self._tests.get(tid)
                if not session:
                    return
                grading = session.get("grading")
                if not isinstance(grading, dict):
                    grading = {}
                    session["grading"] = grading
                pq = grading.get("per_question")
                if not isinstance(pq, list):
                    pq = []
                self._ensure_list_length(pq, total, None)
                if not isinstance(pq[index], dict):
                    pq[index] = {"index": index}
                grading["per_question"] = pq
            async with self._tests_lock:
                session = self._tests.get(tid)
                if not session:
                    return
                per_q = session.get("grading", {}).get("per_question")

        # Mark question as processing.
        async with self._tests_lock:
            session = self._tests.get(tid)
            if session and isinstance(session.get("grading"), dict):
                pq = session["grading"].get("per_question")
                if (
                    isinstance(pq, list)
                    and index < len(pq)
                    and isinstance(pq[index], dict)
                ):
                    pq[index]["status"] = "processing"

        q = quiz[index] if index < len(quiz) else None
        student_answer = ""
        if index < len(answers) and isinstance(answers[index], dict):
            student_answer = str(answers[index].get("answer") or "")

        score: Any = None
        fb: str = ""
        status: str = "done"

        if not isinstance(q, dict) or not q.get("truth_answer"):
            score = None
            fb = "Grading unavailable (missing truth_answer)"
            status = "done"
        else:
            question = str(q.get("question") or "")
            truth_answer = str(q.get("truth_answer") or "")
            rubric = q.get("rubric")
            try:
                grade = await asyncio.to_thread(
                    self.llm.grade_answer_vs_ground_truth,
                    question=question,
                    truth_answer=truth_answer,
                    student_answer=student_answer,
                    rubric=rubric,
                )
                score = grade.get("score_total")
                fb = str(grade.get("feedback") or "")
                status = "done"
            except Exception as e:
                score = None
                fb = f"Grading failed: {e}"
                status = "failed"

        finished_at = datetime.now().isoformat(timespec="seconds")
        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return
            grading = session.get("grading")
            if not isinstance(grading, dict):
                grading = {}
                session["grading"] = grading
            pq = grading.get("per_question")
            if not isinstance(pq, list):
                pq = []
            self._ensure_list_length(pq, total, None)
            if not isinstance(pq[index], dict):
                pq[index] = {"index": index}

            pq[index]["status"] = status
            pq[index]["score_total"] = score
            pq[index]["feedback"] = fb
            grading["per_question"] = pq

            # Update summary fields.
            done_scores: list[int] = []
            all_done = True
            for item in pq:
                if not isinstance(item, dict) or str(item.get("status")) not in {
                    "done",
                    "failed",
                }:
                    all_done = False
                sc = item.get("score_total") if isinstance(item, dict) else None
                if isinstance(sc, int):
                    done_scores.append(sc)

            grading["progress"] = sum(
                1
                for item in pq
                if isinstance(item, dict) and str(item.get("status")) == "done"
            )
            grading["average_score"] = (
                (sum(done_scores) / len(done_scores)) if done_scores else None
            )

            if all_done and session.get("submitted"):
                grading["status"] = "done"
                grading["finished_at"] = finished_at
            else:
                grading["status"] = "processing"

        print(f"[grading] done   test_id={tid} index={index} status={status}")

        # Persist to DB by user name and date.
        try:
            nm = str(session.get("name") or "").strip()
            self.db.enregistrerSession(nm, session)
        except Exception as e:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session:
                    session["grading"]["error"] = f"DB save failed: {e}"

    # -------------------------
    # Interface -> Master calls
    # -------------------------
    async def on_results(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Called by Interface when the front-end posts results."""

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"results_{stamp}.json"
        record = {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "payload": payload,
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {"saved_to": str(out_path), "llm_ready": self.llm is not None}

    # -------------------------
    # Master -> Interface calls
    # -------------------------
    async def shutdown(self) -> None:
        await self.interface.stop()

    async def run(self) -> None:
        await self.interface.serve()


async def main() -> None:
    master = Master(
        llm_kwargs={
            "pdf_path": "C:\\Users\\celli\\Documents\\.PIP2026\\PIP-2026-LOTUSim_G5_Fatigue\\LLM\\COLREG-Consolidated-2018.pdf"
        }
    )

    # Start LLM init concurrently so the server can come up immediately.
    master.start_llm_background()

    # Start background grading worker (queued tasks).
    master.start_grading_worker_background()

    await master.run()


if __name__ == "__main__":
    asyncio.run(main())
