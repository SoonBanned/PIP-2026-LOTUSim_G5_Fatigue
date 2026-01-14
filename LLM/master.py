from __future__ import annotations

import asyncio
import json
import random
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


def _import_qcm_class():
    try:
        from qcm import QCM  # type: ignore

        return QCM
    except Exception:
        from .qcm import QCM  # type: ignore

        return QCM


def _import_scenarios_class():
    try:
        from scenarios import Scenarios  # type: ignore

        return Scenarios
    except Exception:
        from .scenarios import Scenarios  # type: ignore

        return Scenarios


@dataclass
class MasterConfig:
    host: str = "127.0.0.1"
    port: int = 5001
    results_dir: str | Path = Path(__file__).resolve().parent / "results"
    llm_model: str = "llama3.1_fine"
    db_path: str | Path = Path(__file__).resolve().parent / "users_db.json"
    grading_queue_size: int = 32
    qcm_path: str | Path = Path(__file__).resolve().parent / "qcm_final.json"
    scenarios_csv_path: str | Path = (
        Path(__file__).resolve().parent / "colreg_v9_split.csv"
    )
    extra_qcm_count: int = 4
    extra_scenario_count: int = 4


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

        QCM = _import_qcm_class()
        Scenarios = _import_scenarios_class()
        self.qcm = QCM(json_path=self.config.qcm_path)
        self.scenarios = Scenarios(csv_path=self.config.scenarios_csv_path)

        self._tests_lock = asyncio.Lock()
        self._tests: dict[str, dict[str, Any]] = {}

        self._prefetch_lock = asyncio.Lock()
        self._prefetch_tasks: dict[str, asyncio.Task[None]] = {}

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

        # Blocking path: generate quiz now.
        self.start_grading_worker_background()
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

    async def prefetch_test(
        self, *, name: str, n: int = 4, difficulty: str = "easy"
    ) -> Dict[str, Any]:
        """Start quiz generation in the background and return immediately.

        The front-end can poll get_test_status(test_id) until status=="ready".
        """

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
            "quiz_generation": {
                "status": "creating",  # creating | ready | failed
                "error": None,
                "difficulty": str(difficulty),
                "n": int(n),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
            },
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

        # Ensure background tasks exist.
        self.start_llm_background()
        self.start_grading_worker_background()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return {"ok": False, "error": "No running event loop"}

        async with self._prefetch_lock:
            task = loop.create_task(
                self._build_quiz_for_test(
                    test_id=test_id, n=int(n), difficulty=str(difficulty)
                )
            )
            self._prefetch_tasks[test_id] = task

        return {
            "ok": True,
            "test_id": test_id,
            "status": "creating",
            "difficulty": str(difficulty),
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }

    async def get_test_status(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
        if not session:
            return {"ok": False, "error": "Unknown test_id"}

        qg = session.get("quiz_generation")
        if not isinstance(qg, dict):
            # Tests created by the old blocking endpoint are already ready.
            quiz = session.get("quiz") or []
            total = len(quiz) if isinstance(quiz, list) else 0
            return {
                "ok": True,
                "test_id": tid,
                "status": "ready" if total else "creating",
                "total": total,
                "llm_ready": self.llm is not None,
                "llm_error": self.llm_error,
            }

        quiz = session.get("quiz") or []
        total = len(quiz) if isinstance(quiz, list) else 0
        return {
            "ok": True,
            "test_id": tid,
            "status": qg.get("status"),
            "error": qg.get("error"),
            "total": total,
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }

    async def update_test_meta(
        self, *, test_id: str, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return {"ok": False, "error": "Unknown test_id"}
            existing = session.get("meta")
            if not isinstance(existing, dict):
                existing = {}
            for k, v in (meta or {}).items():
                existing[str(k)] = v
            session["meta"] = existing
        return {"ok": True, "test_id": tid}

    def _build_test_items(self, *, n_per_type: int) -> list[dict[str, Any]]:
        """Build a mixed test: n_per_type QCM + n_per_type scenario questions."""
        n_per_type = max(1, int(n_per_type))
        items: list[dict[str, Any]] = []
        items.extend(self.qcm.sample(n=n_per_type))
        items.extend(self.scenarios.sample(n=n_per_type))
        random.shuffle(items)
        return items

    async def _build_quiz_for_test(
        self, *, test_id: str, n: int, difficulty: str
    ) -> None:
        tid = str(test_id)
        try:
            await self.ensure_llm_ready()
            quiz_items = await self._generate_quiz(n=int(n), difficulty=str(difficulty))

            async with self._tests_lock:
                session = self._tests.get(tid)
                if not session:
                    return
                session["quiz"] = quiz_items
                total = len(quiz_items)
                session["grading"]["total"] = total
                session["answers"] = [None] * total
                session["grading"]["per_question"] = [
                    {
                        "index": i,
                        "status": "not_answered",
                        "score_total": None,
                        "feedback": None,
                    }
                    for i in range(total)
                ]
                qg = session.get("quiz_generation")
                if isinstance(qg, dict):
                    qg["status"] = "ready"
                    qg["error"] = None
                    qg["finished_at"] = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session:
                    qg = session.get("quiz_generation")
                    if isinstance(qg, dict):
                        qg["status"] = "failed"
                        qg["error"] = str(e)
                        qg["finished_at"] = datetime.now().isoformat(timespec="seconds")

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

        q_type_for_enqueue: str = "open"

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

            q_item = quiz[i] if isinstance(quiz[i], dict) else {"type": "open"}
            q_type = str(q_item.get("type") or payload.get("type") or "open")
            q_type_for_enqueue = q_type

            answers = session.get("answers")
            if not isinstance(answers, list):
                answers = []
            self._ensure_list_length(answers, total, None)

            # Persist the answer (UI sends {answer, timeSpent, keystrokes, question?}).
            ans_obj = {
                "question": payload.get("question"),
                "type": q_type,
                "answer": payload.get("answer"),
                "selected_index": payload.get("selected_index"),
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
            per_q[i]["type"] = q_type

            if q_type == "qcm":
                # Grade instantly.
                try:
                    selected_index = payload.get("selected_index")
                    sel = int(selected_index) if selected_index is not None else -1
                except Exception:
                    sel = -1
                try:
                    correct_index = int(q_item.get("correct_index", -1))
                except Exception:
                    correct_index = -1

                is_correct = sel == correct_index
                per_q[i]["status"] = "done"
                per_q[i]["score_total"] = 10 if is_correct else 0

                correct_text = None
                choices = q_item.get("choices") if isinstance(q_item, dict) else None
                if isinstance(choices, list) and 0 <= correct_index < len(choices):
                    correct_text = choices[correct_index]
                per_q[i]["feedback"] = (
                    "Correct."
                    if is_correct
                    else (
                        "Incorrect."
                        + (f" Réponse correcte: {correct_text}" if correct_text else "")
                    )
                )
            else:
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

            # If we graded instantly (QCM), update summary fields now.
            if q_type == "qcm":
                done_scores: list[int] = []
                done_count = 0
                all_done = True
                for item in per_q:
                    if not isinstance(item, dict) or str(item.get("status")) not in {
                        "done",
                        "failed",
                    }:
                        all_done = False
                    if isinstance(item, dict) and str(item.get("status")) == "done":
                        done_count += 1
                    sc = item.get("score_total") if isinstance(item, dict) else None
                    if isinstance(sc, int):
                        done_scores.append(sc)
                session["grading"]["progress"] = done_count
                session["grading"]["average_score"] = (
                    (sum(done_scores) / len(done_scores)) if done_scores else None
                )
                if all_done and session.get("submitted"):
                    session["grading"]["status"] = "done"
                    session["grading"]["finished_at"] = datetime.now().isoformat(
                        timespec="seconds"
                    )

        if q_type_for_enqueue == "qcm":
            return {"ok": True, "test_id": tid, "index": i, "status": "done"}

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

        raw_item = quiz[i]
        item = (
            raw_item
            if isinstance(raw_item, dict)
            else {"type": "open", "question": str(raw_item)}
        )

        q_type = str(item.get("type") or "open")

        if q_type == "scenario":
            title = str(item.get("title") or "").strip()
            scenario = str(item.get("scenario") or "").strip()
            question = (title + "\n\n" + scenario).strip() if title else scenario
            img = str(item.get("image") or "").strip()
            image = f"/images_v9/{img}" if img else None
            choices = None
        elif q_type == "qcm":
            question = str(item.get("question") or "").strip()
            image = None
            choices = (
                item.get("choices") if isinstance(item.get("choices"), list) else []
            )
        else:
            question = str(item.get("question") or "").strip()
            image = item.get("image") if isinstance(item.get("image"), str) else None
            choices = (
                item.get("choices") if isinstance(item.get("choices"), list) else None
            )
        return {
            "ok": True,
            "done": False,
            "test_id": tid,
            "index": i,
            "total": total,
            "type": q_type,
            "question": question,
            "image": image,
            "choices": choices,
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
                q_item = quiz[i] if isinstance(quiz, list) and i < len(quiz) else None
                if isinstance(q_item, dict) and str(q_item.get("type") or "") == "qcm":
                    continue
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
            "meta": session.get("meta") or {},
            "answers": session.get("answers") or [],
            "quiz": [
                {
                    "type": q.get("type"),
                    "question": (
                        (
                            str(q.get("title") or "").strip()
                            + "\n\n"
                            + str(q.get("scenario") or "").strip()
                        ).strip()
                        if str(q.get("type") or "") == "scenario"
                        else q.get("question")
                    ),
                }
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
            out.append(
                {
                    "type": "open",
                    "question": base[i % len(base)],
                    "difficulty": "easy",
                    "truth_answer": "",
                    "rubric": None,
                }
            )
        return out

    @staticmethod
    def _difficulty_plan(n: int) -> list[str]:
        """Return per-question difficulties.

        Requirement: for a 4-question quiz -> 2 easy, 1 medium, 1 hard.
        """
        n = max(1, int(n))
        if n == 4:
            return ["easy", "easy", "medium", "hard"]

        # Fallback for other sizes: keep it simple and deterministic-ish.
        out: list[str] = []
        for i in range(n):
            if i < max(1, n // 2):
                out.append("easy")
            elif i < max(2, (3 * n) // 4):
                out.append("medium")
            else:
                out.append("hard")
        return out

    async def _generate_quiz(
        self, *, n: int = 4, difficulty: str = "easy"
    ) -> list[dict[str, Any]]:
        n = max(1, int(n))
        difficulty = str(difficulty or "easy")

        llm_items: list[dict[str, Any]]
        if self.llm is None:
            llm_items = self._fallback_quiz(n)
        else:
            # Ignore the single difficulty string and apply the requested distribution.
            difficulty_lists = self._difficulty_plan(n)
            last_err: Exception | None = None
            for attempt in range(1, 4):
                try:
                    items = await asyncio.to_thread(
                        self.llm.generate_quizz, n, difficulty_lists
                    )
                    if not isinstance(items, list) or not items:
                        raise RuntimeError("LLM returned empty quiz")
                    out: list[dict[str, Any]] = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        it2 = dict(it)
                        it2.setdefault("type", "open")
                        out.append(it2)
                    if not out:
                        raise RuntimeError("LLM returned no valid items")
                    llm_items = out
                    break
                except Exception as e:
                    last_err = e
                    self.llm_error = f"Quiz generation failed: {e}"
                    print(self.llm_error)
                    if attempt < 3:
                        print("Retrying quiz generation")
                        await asyncio.sleep(0.2)
            else:
                if last_err is not None:
                    self.llm_error = (
                        f"Quiz generation failed (fallback used): {last_err}"
                    )
                llm_items = self._fallback_quiz(n)

        # Requirement: keep LLM questions as-is, but add 4 QCM + 4 scenarios.
        extra_items: list[dict[str, Any]] = []
        try:
            qcm_n = max(0, int(getattr(self.config, "extra_qcm_count", 4)))
        except Exception:
            qcm_n = 4
        try:
            sc_n = max(0, int(getattr(self.config, "extra_scenario_count", 4)))
        except Exception:
            sc_n = 4

        if qcm_n:
            try:
                extra_items.extend(self.qcm.sample(n=qcm_n))
            except Exception as e:
                print(f"QCM sampling failed: {e}")
        if sc_n:
            try:
                extra_items.extend(self.scenarios.sample(n=sc_n))
            except Exception as e:
                print(f"Scenario sampling failed: {e}")

        if extra_items:
            random.shuffle(extra_items)
            return llm_items + extra_items
        return llm_items

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

        q_type = str(q.get("type") or "open") if isinstance(q, dict) else "open"
        score: Any = None
        fb: str = ""
        status: str = "done"

        await self.ensure_llm_ready()
        if self.llm is None:
            score = None
            fb = self.llm_error or "LLM not available"
            status = "failed"
        elif q_type == "scenario" and isinstance(q, dict):
            scenario_text = str(q.get("scenario") or q.get("question") or "")
            reference_application = q.get("reference_application")
            source_id = q.get("id")
            try:
                grade = await asyncio.to_thread(
                    self.llm.grade_answer_vs_rule_application,
                    scenario=scenario_text,
                    student_answer=student_answer,
                    source_id=source_id,
                    reference_application=reference_application,
                )
                if isinstance(grade, dict):
                    score = grade.get("score")
                    if score is None:
                        score = grade.get("score_total")
                    fb = str(grade.get("feedback") or "")
                    status = "done"
                else:
                    score = None
                    fb = "Unexpected grade format"
                    status = "failed"
            except Exception as e:
                score = None
                fb = f"Scenario grading failed: {e}"
                status = "failed"
        elif isinstance(q, dict) and str(q.get("truth_answer") or "").strip():
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
                if isinstance(grade, dict):
                    score = grade.get("score_total")
                    fb = str(grade.get("feedback") or "")
                    status = "done"
                else:
                    score = None
                    fb = "Unexpected grade format"
                    status = "failed"
            except Exception as e:
                score = None
                fb = f"Grading failed: {e}"
                status = "failed"
        else:
            score = None
            fb = "Grading unavailable (missing truth_answer)"
            status = "done"

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
