from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_import_cv2():
    import cv2  # type: ignore

    return cv2


def _try_open_camera(cv2: Any, index: int) -> bool:
    backends: list[int | None] = []
    if os.name == "nt":
        backends.extend(
            [
                getattr(cv2, "CAP_DSHOW", None),
                getattr(cv2, "CAP_MSMF", None),
            ]
        )
    backends.append(getattr(cv2, "CAP_ANY", None))

    for backend in backends:
        if backend is None:
            continue
        cap = None
        try:
            try:
                cap = cv2.VideoCapture(int(index), int(backend))
            except Exception:
                cap = cv2.VideoCapture(int(index))

            if not cap or not cap.isOpened():
                continue

            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return True
        except Exception:
            continue
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    return False


try:
    from .video_session import VideoConfig, VideoSession, VideoSummary
except Exception:
    from video_session import VideoConfig, VideoSession, VideoSummary  # type: ignore


class VideoHandler:
    """Owns VideoSession objects (one per test_id).

    Designed to be called from Master (more video logic will be added later).
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        default_config: Optional[VideoConfig] = None,
    ) -> None:
        self.model_path = model_path
        self.default_config = default_config or VideoConfig()

        self._lock = threading.Lock()
        self._sessions: Dict[str, VideoSession] = {}

    def list_cameras(self, *, max_index: int = 6) -> Dict[str, Any]:
        """Best-effort scan of numeric camera indices usable by OpenCV."""

        cv2 = _safe_import_cv2()
        found: List[int] = []
        for i in range(max(0, int(max_index))):
            try:
                if _try_open_camera(cv2, int(i)):
                    found.append(int(i))
            except Exception:
                continue

        return {"ok": True, "cameras": found}

    def create(self, *, test_id: str, camera_index: int) -> Dict[str, Any]:
        tid = str(test_id)
        cam = int(camera_index)
        with self._lock:
            if tid in self._sessions:
                # keep existing; allow updating only if idle
                return {"ok": True, "status": "exists"}

            self._sessions[tid] = VideoSession(
                camera_index=cam,
                config=self.default_config,
                model_path=self.model_path,
            )

        return {"ok": True, "status": "created"}

    def start(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        with self._lock:
            sess = self._sessions.get(tid)
        if sess is None:
            return {"ok": False, "error": "No video session"}
        return sess.start()

    def stop(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        with self._lock:
            sess = self._sessions.get(tid)
        if sess is None:
            return {"ok": False, "error": "No video session"}
        summary = sess.stop()
        return {"ok": True, "summary": summary.__dict__}

    def status(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        with self._lock:
            sess = self._sessions.get(tid)
        if sess is None:
            return {"ok": False, "error": "No video session"}
        return sess.status()

    def get_summary_if_any(self, *, test_id: str) -> Optional[VideoSummary]:
        tid = str(test_id)
        with self._lock:
            sess = self._sessions.get(tid)
        if sess is None:
            return None
        # Safe: call stop(summary) if already done? don't stop implicitly
        st = sess.status()
        if not st.get("ok"):
            return None
        if st.get("status") in {"completed", "stopped", "failed"}:
            return sess.stop(timeout=0.0)
        return None
