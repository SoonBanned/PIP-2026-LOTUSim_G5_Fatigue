from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from database import Database
except Exception:  # pragma: no cover
    from .database import Database


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


def _import_video_handler_class():
    try:
        from video.handler import VideoHandler  # type: ignore

        return VideoHandler
    except Exception:
        try:
            from .video.handler import VideoHandler  # type: ignore

            return VideoHandler
        except Exception:
            return None


def _import_fatigue_handler_class():
    try:
        from fatigue import FatigueHandler  # type: ignore

        return FatigueHandler
    except Exception:
        try:
            from .fatigue import FatigueHandler  # type: ignore

            return FatigueHandler
        except Exception:
            return None


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

        # Typing/interaction fatigue analysis (non-video).
        self.fatigue_handler: Any | None = None
        FatigueHandler = _import_fatigue_handler_class()
        if FatigueHandler is not None:
            try:
                self.fatigue_handler = FatigueHandler()
            except Exception as e:
                print(f"Fatigue handler init failed: {e}")
                self.fatigue_handler = None

        # Optional video fatigue estimation handler.
        self.video_handler: Any | None = None
        VideoHandler = _import_video_handler_class()
        if VideoHandler is not None:
            try:
                video_dir = Path(__file__).resolve().parent / "video"
                model_path = video_dir / "face_landmarker.task"
                self.video_handler = VideoHandler(model_path=model_path)
            except Exception as e:
                print(f"Video handler init failed: {e}")
                self.video_handler = None

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

    async def get_user_profile(self, *, name: str) -> Dict[str, Any]:
        """Return last-known user meta to improve UX (age + camera prefs)."""
        nm = self._sanitize_name(name)
        if not nm:
            return {"ok": False, "error": "Name is required"}

        try:
            sessions = self.db.recupererResultats(nm)
        except Exception as e:
            return {"ok": False, "error": f"DB error: {e}"}

        last_meta: dict[str, Any] = {}
        if isinstance(sessions, list) and sessions:
            # Newest session is the last one in storage order.
            for s in reversed(sessions):
                if not isinstance(s, dict):
                    continue
                payload = s.get("payload") if isinstance(s.get("payload"), dict) else {}
                meta = (
                    payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                )
                if isinstance(meta, dict) and meta:
                    last_meta = dict(meta)
                    break

        profile = {
            "age": last_meta.get("age"),
            "consecutive_work_hours": last_meta.get("consecutive_work_hours"),
            "tiredness": last_meta.get("tiredness"),
            "camera_index": last_meta.get("camera_index"),
            "camera_device_id": last_meta.get("camera_device_id"),
            "camera_label": last_meta.get("camera_label"),
        }
        return {"ok": True, "name": nm, "profile": profile}

    async def get_user_history(self, *, name: str, limit: int = 5) -> Dict[str, Any]:
        nm = self._sanitize_name(name)
        if not nm:
            return {"ok": False, "error": "Name is required"}
        try:
            raw = self.db.recupererResultats(nm)
        except Exception as e:
            return {"ok": False, "error": f"DB error: {e}"}

        sessions = raw if isinstance(raw, list) else []
        try:
            limit_n = max(0, int(limit))
        except Exception:
            limit_n = 5

        # Newest first
        sessions = list(reversed(sessions))
        if limit_n:
            sessions = sessions[:limit_n]

        out: list[dict[str, Any]] = []
        for s in sessions:
            if not isinstance(s, dict):
                continue
            out.append(
                {
                    "session_id": s.get("session_id"),
                    "date_time": s.get("date_time"),
                    "mental_fatigue": s.get("mental_fatigue"),
                    "physical_fatigue": s.get("physical_fatigue"),
                }
            )

        # Also compute aggregated series for "compare to others" mode (best-effort).
        others_series: list[dict[str, Any]] = []
        try:
            others_series = self.db.serie_autres_utilisateurs(nm)
        except Exception:
            others_series = []

        return {
            "ok": True,
            "name": nm,
            "history": out,
            "others_series": others_series,
        }

    async def get_user_session(self, *, name: str, session_id: str) -> Dict[str, Any]:
        nm = self._sanitize_name(name)
        sid = str(session_id or "").strip()
        if not nm or not sid:
            return {"ok": False, "error": "Missing name/session_id"}
        try:
            sess = self.db.recuperer_session(nm, sid)
        except Exception as e:
            return {"ok": False, "error": f"DB error: {e}"}
        if not sess:
            return {"ok": False, "error": "Session not found"}
        payload = sess.get("payload") if isinstance(sess.get("payload"), dict) else {}
        return {
            "ok": True,
            "session": {
                "session_id": sess.get("session_id"),
                "date_time": sess.get("date_time"),
                "mental_fatigue": sess.get("mental_fatigue"),
                "physical_fatigue": sess.get("physical_fatigue"),
            },
            "payload": payload,
        }

    def _compute_mental_physical_scores(
        self, *, session: dict[str, Any]
    ) -> tuple[float | None, float | None]:
        grading = (
            session.get("grading") if isinstance(session.get("grading"), dict) else {}
        )
        avg = grading.get("average_score") if isinstance(grading, dict) else None
        avg_score: float | None = None
        try:
            if avg is not None:
                avg_score = float(avg)
        except Exception:
            avg_score = None

        # Mental fatigue: invert performance score (0-10) into fatigue (0-100).
        mental: float | None = None
        if avg_score is not None:
            mental = max(0.0, min(100.0, 100.0 - (avg_score / 10.0) * 100.0))

        # Physical fatigue: prefer camera score, fallback to typing fatigue score.
        physical: float | None = None
        video = session.get("video") if isinstance(session.get("video"), dict) else {}
        if isinstance(video, dict):
            v = video.get("fatigue_score_avg")
            if v is None:
                v = video.get("fatigue_score_last")
            try:
                if v is not None:
                    physical = float(v)
            except Exception:
                physical = None

        if physical is None:
            fatigue = (
                session.get("fatigue")
                if isinstance(session.get("fatigue"), dict)
                else {}
            )
            if isinstance(fatigue, dict):
                v = fatigue.get("score_fatigue_global")
                try:
                    if v is not None:
                        physical = float(v)
                except Exception:
                    physical = None

        if physical is not None:
            physical = max(0.0, min(100.0, physical))

        return mental, physical

    @staticmethod
    def _slim_payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
        """Remove heavy per-answer input timing details before persisting to DB."""
        if not isinstance(payload, dict):
            return {}

        out = dict(payload)
        answers = out.get("answers")
        if isinstance(answers, list):
            slim_answers: list[Any] = []
            for a in answers:
                if isinstance(a, dict):
                    aa = dict(a)
                    # Remove potentially huge raw timing payloads.
                    aa.pop("keystrokes", None)
                    aa.pop("keystrokes_raw", None)
                    aa.pop("timings", None)
                    slim_answers.append(aa)
                else:
                    slim_answers.append(a)
            out["answers"] = slim_answers

        # If the backend ever included raw fatigue inputs, drop them.
        fatigue = out.get("fatigue")
        if isinstance(fatigue, dict):
            ff = dict(fatigue)
            ff.pop("raw", None)
            ff.pop("samples", None)
            ff.pop("per_answer", None)
            out["fatigue"] = ff

        return out

    async def _persist_session_if_ready(self, *, test_id: str) -> None:
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
            if not session:
                return
            if session.get("_persisted"):
                return
            grading = (
                session.get("grading")
                if isinstance(session.get("grading"), dict)
                else {}
            )
            if not isinstance(grading, dict) or str(grading.get("status")) != "done":
                return
            if not session.get("submitted"):
                return

            nm = str(session.get("name") or "").strip()
            if not nm:
                return

        # Build a stable payload for later review and strip heavy timing data.
        payload_full = await self.get_grading_result(test_id=tid)
        payload = self._slim_payload_for_storage(payload_full)

        # Compute scores from the finalized payload (not from raw inputs).
        mental: float | None = None
        physical: float | None = None
        try:
            grading_out = (
                payload.get("grading")
                if isinstance(payload.get("grading"), dict)
                else {}
            )
            avg = (
                grading_out.get("average_score")
                if isinstance(grading_out, dict)
                else None
            )
            avg_score = float(avg) if avg is not None else None
            if avg_score is not None:
                mental = max(0.0, min(100.0, 100.0 - (avg_score / 10.0) * 100.0))
        except Exception:
            mental = None

        # Physical fatigue: camera (if available) else fatigue.py score.
        try:
            video_out = (
                payload.get("video") if isinstance(payload.get("video"), dict) else {}
            )
            if isinstance(video_out, dict):
                v = video_out.get("fatigue_score_avg")
                if v is None:
                    v = video_out.get("fatigue_score_last")
                if v is not None:
                    physical = float(v)
        except Exception:
            physical = None

        if physical is None:
            try:
                fatigue_out = (
                    payload.get("fatigue")
                    if isinstance(payload.get("fatigue"), dict)
                    else {}
                )
                if isinstance(fatigue_out, dict):
                    v = fatigue_out.get("score_fatigue_global")
                    if v is not None:
                        physical = float(v)
            except Exception:
                physical = None

        if physical is not None:
            physical = max(0.0, min(100.0, physical))
        date_time = str(
            (session.get("date") or grading.get("finished_at") or "")
        ).strip() or datetime.now().isoformat(timespec="seconds")
        try:
            sid = self.db.ajouter_session(
                nm,
                date_time=date_time,
                mental_fatigue=mental,
                physical_fatigue=physical,
                payload=payload,
            )
        except Exception as e:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session and isinstance(session.get("grading"), dict):
                    session["grading"]["error"] = f"DB save failed: {e}"
            return

        async with self._tests_lock:
            session = self._tests.get(tid)
            if session:
                session["_persisted"] = True
                session["session_id"] = sid

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

        # Start fatigue session now (best-effort).
        if self.fatigue_handler is not None:
            try:
                self.fatigue_handler.start(test_id=test_id)
            except Exception:
                pass

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
                "attempt": 0,
                "max_attempts": 5,
                "last_error": None,
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

        # Start fatigue session now (best-effort).
        if self.fatigue_handler is not None:
            try:
                self.fatigue_handler.start(test_id=test_id)
            except Exception:
                pass

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
            "attempt": qg.get("attempt"),
            "max_attempts": qg.get("max_attempts"),
            "last_error": qg.get("last_error"),
            "total": total,
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }

    async def _generate_quiz_with_status(
        self, *, test_id: str, n: int = 4, difficulty: str = "easy"
    ) -> list[dict[str, Any]]:
        """Generate quiz items while updating session.quiz_generation attempt fields."""

        tid = str(test_id)

        n = max(1, int(n))
        difficulty = str(difficulty or "easy")

        # Ensure LLM is ready first (can dominate the wait time).
        await self.ensure_llm_ready()
        if self.llm is None:
            return await self._generate_quiz(n=n, difficulty=difficulty)

        difficulty_lists = self._difficulty_plan(n)
        last_err: Exception | None = None

        for attempt in range(1, 4):
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session:
                    qg = session.get("quiz_generation")
                    if isinstance(qg, dict):
                        qg["attempt"] = attempt
                        qg["max_attempts"] = 3
                        qg["last_error"] = None

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
                return self._append_extras_and_shuffle(out)
            except Exception as e:
                last_err = e
                async with self._tests_lock:
                    session = self._tests.get(tid)
                    if session:
                        qg = session.get("quiz_generation")
                        if isinstance(qg, dict):
                            qg["last_error"] = str(e)
                self.llm_error = f"Quiz generation failed: {e}"
                print(self.llm_error)
                if attempt < 3:
                    print("Retrying quiz generation")
                    await asyncio.sleep(0.2)

        if last_err is not None:
            self.llm_error = f"Quiz generation failed (fallback used): {last_err}"
        return self._append_extras_and_shuffle(self._fallback_quiz(n))

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

            # If the UI provides a numeric OpenCV camera index, create a video session now.
            if self.video_handler is not None:
                cam_idx = existing.get("camera_index")
                if cam_idx is not None:
                    try:
                        self.video_handler.create(
                            test_id=tid, camera_index=int(cam_idx)
                        )
                    except Exception as e:
                        print(f"Video create failed: {e}")
        return {"ok": True, "test_id": tid}

    def _build_test_items(self, *, n_per_type: int) -> list[dict[str, Any]]:
        """Build a mixed test: n_per_type QCM + n_per_type scenario questions."""
        n_per_type = max(1, int(n_per_type))
        items: list[dict[str, Any]] = []
        items.extend(self.qcm.sample(n=n_per_type))
        items.extend(self.scenarios.sample(n=n_per_type))
        random.shuffle(items)
        return items

    def _append_extras_and_shuffle(
        self, base_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Append dataset questions (QCM + scenarios) and shuffle for the user."""

        out: list[dict[str, Any]] = list(base_items or [])

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
                out.extend(self.qcm.sample(n=qcm_n))
            except Exception as e:
                print(f"QCM sampling failed: {e}")
        if sc_n:
            try:
                out.extend(self.scenarios.sample(n=sc_n))
            except Exception as e:
                print(f"Scenario sampling failed: {e}")

        random.shuffle(out)
        return out

    async def _build_quiz_for_test(
        self, *, test_id: str, n: int, difficulty: str
    ) -> None:
        tid = str(test_id)
        try:
            quiz_items = await self._generate_quiz_with_status(
                test_id=tid, n=int(n), difficulty=str(difficulty)
            )

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

            # Start camera recording once the quiz is ready (if configured).
            await self._maybe_start_video_for_test(test_id=tid)
        except Exception as e:
            async with self._tests_lock:
                session = self._tests.get(tid)
                if session:
                    qg = session.get("quiz_generation")
                    if isinstance(qg, dict):
                        qg["status"] = "failed"
                        qg["error"] = str(e)
                        qg["finished_at"] = datetime.now().isoformat(timespec="seconds")

    async def _maybe_start_video_for_test(self, *, test_id: str) -> None:
        if self.video_handler is None:
            return
        tid = str(test_id)
        async with self._tests_lock:
            session = self._tests.get(tid)
            meta = session.get("meta") if isinstance(session, dict) else None
        cam_idx = None
        if isinstance(meta, dict) and meta.get("camera_index") is not None:
            cam_idx = meta.get("camera_index")

        if cam_idx is None:
            return

        try:
            self.video_handler.create(test_id=tid, camera_index=int(cam_idx))
        except Exception:
            pass

        try:
            self.video_handler.start(test_id=tid)
        except Exception as e:
            print(f"Video start failed: {e}")

    async def _maybe_stop_video_for_test(
        self, *, test_id: str
    ) -> Dict[str, Any] | None:
        if self.video_handler is None:
            return None
        tid = str(test_id)
        try:
            res = self.video_handler.stop(test_id=tid)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        async with self._tests_lock:
            session = self._tests.get(tid)
            if session is not None and isinstance(session, dict):
                session["video"] = res.get("summary") if isinstance(res, dict) else res
        return res

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

        # Feed timing/keystroke data to fatigue module (best-effort).
        if self.fatigue_handler is not None:
            try:
                self.fatigue_handler.ingest_answer(test_id=tid, answer=ans_obj)
            except Exception:
                pass

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
            extra = {"title": title, "description": scenario}
        elif q_type == "qcm":
            question = str(item.get("question") or "").strip()
            image = None
            choices = (
                item.get("choices") if isinstance(item.get("choices"), list) else []
            )
            extra = {}
        else:
            question = str(item.get("question") or "").strip()
            image = item.get("image") if isinstance(item.get("image"), str) else None
            choices = (
                item.get("choices") if isinstance(item.get("choices"), list) else None
            )
            extra = {}
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
            **extra,
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

        # Stop camera recording on final submission (best-effort).
        await self._maybe_stop_video_for_test(test_id=tid)

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
        fatigue: dict[str, Any] = {}
        if self.fatigue_handler is not None and isinstance(grading, dict):
            try:
                answers = session.get("answers") or []
                if not isinstance(answers, list):
                    answers = []
                fres = self.fatigue_handler.finalize(
                    test_id=tid, grading=grading, answers=answers
                )
                if isinstance(fres, dict) and fres.get("ok"):
                    fatigue = fres.get("fatigue") or {}
            except Exception:
                fatigue = {}
        return {
            "ok": True,
            "test_id": tid,
            "status": grading.get("status"),
            "grading": grading,
            "meta": session.get("meta") or {},
            "video": session.get("video") or {},
            "fatigue": fatigue,
            "totalTestTime": session.get("totalTestTime"),
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
                    "difficulty": q.get("difficulty"),
                    "image": (
                        f"/images_v9/{str(q.get('image') or '').strip()}"
                        if str(q.get("type") or "") == "scenario"
                        and str(q.get("image") or "").strip()
                        else q.get("image")
                    ),
                    "choices": (
                        q.get("choices") if isinstance(q.get("choices"), list) else None
                    ),
                    "correct_index": q.get("correct_index"),
                    "truth_answer": q.get("truth_answer"),
                }
                for q in (session.get("quiz") or [])
                if isinstance(q, dict)
            ],
        }

    # -------------------------
    # Video API
    # -------------------------
    async def list_video_cameras(self, *, max_index: int = 6) -> Dict[str, Any]:
        if self.video_handler is None:
            return {"ok": False, "error": "Video handler unavailable"}
        try:
            return self.video_handler.list_cameras(max_index=int(max_index))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def start_video(self, *, test_id: str) -> Dict[str, Any]:
        if self.video_handler is None:
            return {"ok": False, "error": "Video handler unavailable"}
        tid = str(test_id)
        # Ensure session exists + has camera_index.
        await self._maybe_start_video_for_test(test_id=tid)
        try:
            return self.video_handler.status(test_id=tid)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def stop_video(self, *, test_id: str) -> Dict[str, Any]:
        res = await self._maybe_stop_video_for_test(test_id=str(test_id))
        return res or {"ok": False, "error": "Video handler unavailable"}

    async def get_video_status(self, *, test_id: str) -> Dict[str, Any]:
        if self.video_handler is None:
            return {"ok": False, "error": "Video handler unavailable"}
        try:
            return self.video_handler.status(test_id=str(test_id))
        except Exception as e:
            return {"ok": False, "error": str(e)}

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
        return self._append_extras_and_shuffle(llm_items)

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

        # Persist once when fully done.
        try:
            await self._persist_session_if_ready(test_id=tid)
        except Exception:
            pass

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
