from __future__ import annotations

import threading
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_import_cv2():
    import cv2  # type: ignore

    return cv2


def _open_capture(cv2: Any, camera_index: int, *, fps: int) -> Any | None:
    """Open a camera robustly.

    On Windows, MSMF can fail with:
      CvCapture_MSMF::grabFrame ... Error: -1072875772
    so we prefer DirectShow first.
    """

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
        cap = None
        try:
            if backend is None:
                continue
            try:
                cap = cv2.VideoCapture(int(camera_index), int(backend))
            except Exception:
                cap = cv2.VideoCapture(int(camera_index))

            if not cap or not cap.isOpened():
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                continue

            # Best-effort FPS hint
            try:
                cap.set(cv2.CAP_PROP_FPS, float(fps))
            except Exception:
                pass

            # Warmup reads
            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)

            # Warmup failed; close and try next backend.
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            continue

    return None


def _safe_import_numpy():
    import numpy as np  # type: ignore

    return np


def _safe_import_mediapipe():
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore

    return mp, python, vision


def _try_import_scipy_stats():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


@dataclass
class VideoConfig:
    window_size_seconds: int = 30
    fps: int = 30
    ear_threshold: float = 0.2
    analysis_block_seconds: int = 30
    max_blocks: int = 3
    calibration_seconds: int = 10
    # weights kept from your Estimation_facile_v2.py
    weights: Dict[str, float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "perclos": 0.35,
                "blink_duration": 0.25,
                "ear_mean": 0.20,
                "gaze_diversity": 0.10,
                "facial_activity": 0.10,
            }


class FatigueIndicators:
    def __init__(self, window_size: int = 30, fps: int = 30):
        np = _safe_import_numpy()
        from collections import deque

        self.window_size = int(window_size)
        self.fps = int(fps)
        self.buffer_size = self.window_size * self.fps

        self.ear_buffer = deque(maxlen=self.buffer_size)
        self.blink_events = deque(maxlen=100)
        self.gaze_positions = deque(maxlen=self.buffer_size)
        self.landmark_positions = deque(maxlen=self.buffer_size)

        self.blink_in_progress = False
        self.blink_start_frame = 0
        self.ear_threshold = 0.2

        self._np = np

    @staticmethod
    def calculate_ear(eye_landmarks) -> float:
        np = _safe_import_numpy()
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def update_ear(self, left_eye, right_eye, frame_idx: int) -> float:
        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear_mean = (ear_left + ear_right) / 2.0
        self.ear_buffer.append(ear_mean)

        if ear_mean < self.ear_threshold:
            if not self.blink_in_progress:
                self.blink_in_progress = True
                self.blink_start_frame = int(frame_idx)
        else:
            if self.blink_in_progress:
                blink_duration = (int(frame_idx) - int(self.blink_start_frame)) / float(
                    self.fps
                )
                self.blink_events.append(float(blink_duration))
                self.blink_in_progress = False

        return float(ear_mean)

    def calculate_perclos(self) -> float:
        if len(self.ear_buffer) == 0:
            return 0.0
        closed_frames = sum(1 for ear in self.ear_buffer if ear < self.ear_threshold)
        return float(closed_frames / len(self.ear_buffer))

    def calculate_blink_statistics(self) -> Dict[str, float]:
        np = self._np
        durations = list(self.blink_events)
        count = len(durations)
        if count == 0:
            return {
                "mean_duration": 0.0,
                "std_duration": 0.0,
                "frequency": 0.0,
                "count": 0.0,
            }

        mean_duration = float(np.mean(durations))
        std_duration = float(np.std(durations))

        time_window = min(
            len(self.ear_buffer) / float(self.fps), float(self.window_size)
        )
        frequency = (count / time_window) * 60.0 if time_window > 0 else 0.0

        return {
            "mean_duration": mean_duration,
            "std_duration": std_duration,
            "frequency": float(frequency),
            "count": float(count),
        }

    def calculate_ear_statistics(self) -> Dict[str, float]:
        np = self._np
        if len(self.ear_buffer) < 10:
            return {"mean": 0.0, "std": 0.0, "slope": 0.0}

        ear_array = np.array(self.ear_buffer)
        mean_ear = float(np.mean(ear_array))
        std_ear = float(np.std(ear_array))

        x = np.arange(len(ear_array))
        stats = _try_import_scipy_stats()
        if stats is not None:
            slope, _, _, _, _ = stats.linregress(x, ear_array)
            slope_val = float(slope)
        else:
            # Fallback: simple linear fit
            slope_val = float(np.polyfit(x, ear_array, 1)[0])

        return {"mean": mean_ear, "std": std_ear, "slope": slope_val}

    def update_gaze_position(self, gaze_vector) -> None:
        self.gaze_positions.append(gaze_vector)

    def calculate_gaze_dispersion(self) -> float:
        np = self._np
        if len(self.gaze_positions) < 10:
            return 0.0

        positions = np.array(self.gaze_positions)
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        total_variance = float(np.sqrt(var_x + var_y))
        normalized_variance = min(total_variance / 0.1, 1.0)
        return float(normalized_variance)

    def update_facial_landmarks(self, landmarks) -> None:
        self.landmark_positions.append(landmarks.flatten())

    def calculate_facial_activity(self) -> float:
        np = self._np
        if len(self.landmark_positions) < 2:
            return 0.0

        positions = np.array(self.landmark_positions)
        diffs = np.diff(positions, axis=0)
        movements = np.linalg.norm(diffs, axis=1)
        energy = float(np.mean(movements))
        normalized_energy = min(energy / 0.02, 1.0)
        return float(normalized_energy)


class FatigueEstimator:
    def __init__(
        self,
        *,
        config: Optional[VideoConfig] = None,
        model_path: str | Path | None = None,
    ):
        cv2 = _safe_import_cv2()
        np = _safe_import_numpy()
        mp, python, vision = _safe_import_mediapipe()
        from collections import deque
        import os

        self._cv2 = cv2
        self._np = np
        self._mp = mp
        self._vision = vision

        cfg = config or VideoConfig()
        self.config = {
            "window_size": int(cfg.window_size_seconds),
            "fps": int(cfg.fps),
            "ear_threshold": float(cfg.ear_threshold),
            "weights": dict(cfg.weights),
        }

        mp_model_path = None
        if model_path is not None:
            mp_model_path = str(model_path)
        else:
            mp_model_path = "face_landmarker.task"
            if not os.path.exists(mp_model_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mp_model_path = os.path.join(current_dir, "face_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=mp_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        self.start_time = time.time()

        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473, 474, 475]

        self.indicators = FatigueIndicators(
            window_size=self.config["window_size"], fps=self.config["fps"]
        )

        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_limit = 10 * self.config["fps"]
        self.ear_baseline = 0.3
        self.calibration_values: List[float] = []

        self.current_block_scores: List[float] = []
        self.block_results: List[float] = []

        self.analysis_duration = 30 * self.config["fps"]
        self.max_blocks = 3

        self.total_fatigue_history = deque(maxlen=self.config["fps"] * 30)
        self.global_fatigue_percentage = 0.0

        self.history: List[Dict[str, Any]] = []
        self.frame_count = 0
        self.analysis_started_frame = 0

    def extract_eye_landmarks(
        self, face_landmarks, image_shape: Tuple[int, int], eye_indices: List[int]
    ):
        np = self._np
        h, w = image_shape
        coords = []
        for idx in eye_indices:
            landmark = face_landmarks[idx]
            coords.append([landmark.x * w, landmark.y * h])
        return np.array(coords)

    def estimate_gaze_direction(self, face_landmarks, image_shape: Tuple[int, int]):
        np = self._np
        iris_coords = []
        for idx in self.IRIS_INDICES[:4]:
            landmark = face_landmarks[idx]
            iris_coords.append([landmark.x, landmark.y])
        iris_center = np.mean(iris_coords, axis=0)
        return iris_center

    def calculate_fatigue_score(self) -> Tuple[float, Dict[str, float]]:
        np = self._np

        perclos = self.indicators.calculate_perclos()
        blink_stats = self.indicators.calculate_blink_statistics()
        ear_stats = self.indicators.calculate_ear_statistics()
        gaze_dispersion = self.indicators.calculate_gaze_dispersion()
        facial_activity = self.indicators.calculate_facial_activity()

        custom_threshold = self.ear_baseline * 0.82
        self.indicators.ear_threshold = custom_threshold

        norm_ear = 1.0 - float(np.clip(ear_stats["mean"] / self.ear_baseline, 0, 1))
        norm_perclos = perclos * 1.5
        norm_blink = float(np.clip(blink_stats["mean_duration"] / 0.4, 0, 1))
        norm_gaze = 1.0 - gaze_dispersion
        norm_facial = 1.0 - facial_activity

        weights = self.config["weights"]
        score = (
            norm_perclos * weights["perclos"]
            + norm_blink * weights["blink_duration"]
            + norm_ear * weights["ear_mean"]
            + norm_gaze * weights["gaze_diversity"]
            + norm_facial * weights["facial_activity"]
        )

        score = float(np.power(score, 0.8) * 100.0)

        self.total_fatigue_history.append(score)
        self.global_fatigue_percentage = float(np.mean(self.total_fatigue_history))

        indicators_dict = {
            "perclos": float(perclos),
            "blink_mean_duration": float(blink_stats["mean_duration"]),
            "blink_frequency": float(blink_stats["frequency"]),
            "blink_count": float(blink_stats["count"]),
            "ear_mean": float(ear_stats["mean"]),
            "ear_baseline": float(self.ear_baseline),
            "gaze_dispersion": float(gaze_dispersion),
            "facial_activity": float(facial_activity),
            "global_fatigue": float(self.global_fatigue_percentage),
            "normalized_perclos": float(norm_perclos),
            "normalized_blink": float(norm_blink),
            "normalized_ear": float(norm_ear),
            "normalized_gaze": float(norm_gaze),
            "normalized_facial": float(norm_facial),
        }

        return score, indicators_dict

    def process_frame(self, frame):
        cv2 = self._cv2
        mp = self._mp
        np = self._np

        self.frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - self.start_time) * 1000)
        results = self.face_mesh.detect_for_video(mp_image, timestamp_ms)

        results_dict: Dict[str, Any] = {"face_detected": False}

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            h, w = frame.shape[:2]

            left_eye = self.extract_eye_landmarks(
                face_landmarks, (h, w), self.LEFT_EYE_INDICES
            )
            right_eye = self.extract_eye_landmarks(
                face_landmarks, (h, w), self.RIGHT_EYE_INDICES
            )

            ear = self.indicators.update_ear(left_eye, right_eye, self.frame_count)

            gaze = self.estimate_gaze_direction(face_landmarks, (h, w))
            self.indicators.update_gaze_position(gaze)

            all_landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])
            self.indicators.update_facial_landmarks(all_landmarks)

            if not self.is_calibrated:
                self.calibration_values.append(float(ear))
                self.calibration_frames += 1
                if self.calibration_frames >= self.calibration_limit:
                    self.ear_baseline = float(
                        np.percentile(self.calibration_values, 90)
                    )
                    self.is_calibrated = True
                    self.analysis_started_frame = self.frame_count
                return {"face_detected": True, "calibrating": True}

            relative_frame = self.frame_count - self.analysis_started_frame
            current_block_idx = relative_frame // self.analysis_duration

            if current_block_idx >= self.max_blocks:
                return {"analysis_complete": True}

            fatigue_score, indicators = self.calculate_fatigue_score()
            self.current_block_scores.append(float(fatigue_score))

            avg_block_score = float(np.mean(self.current_block_scores))

            if relative_frame > 0 and relative_frame % self.analysis_duration == 0:
                self.block_results.append(avg_block_score)
                self.current_block_scores = []

            results_dict = {
                "face_detected": True,
                "fatigue_score": float(fatigue_score),
                "ear": float(ear),
                "timestamp": float(self.frame_count / float(self.config["fps"])),
                **indicators,
            }
            self.history.append(results_dict)

        return results_dict


@dataclass
class VideoSummary:
    ok: bool
    status: str
    camera_index: int
    started_at: float | None
    stopped_at: float | None
    duration_seconds: float | None
    fatigue_score_avg: float | None
    fatigue_score_last: float | None
    fatigue_score_blocks: List[float]
    verdict: str | None
    error: str | None = None


class VideoSession:
    """Run fatigue estimation in a background thread.

    - init(camera_index)
    - start(): begins capturing + computing
    - stop(): stops and returns a score summary

    Keeps the same core logic as Estimation_facile_v2.py:
    - invisible calibration
    - 3 blocks of 30s analysis
    - fatigue score computed per frame
    """

    def __init__(
        self,
        *,
        camera_index: int,
        config: Optional[VideoConfig] = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.camera_index = int(camera_index)
        self.config = config or VideoConfig()
        self.model_path = model_path

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

        self._status: str = "idle"  # idle|running|completed|stopped|failed
        self._error: str | None = None

        self._started_at: float | None = None
        self._stopped_at: float | None = None

        self._estimator: FatigueEstimator | None = None
        self._last_score: float | None = None
        self._completed_by_model: bool = False

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": True, "status": self._status}
            if self._status in {"completed", "stopped"}:
                # do not restart a finished session
                return {"ok": False, "error": f"Video session already {self._status}"}

            self._stop_evt.clear()
            self._status = "running"
            self._error = None
            self._started_at = time.time()
            self._stopped_at = None
            self._completed_by_model = False

            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        return {"ok": True, "status": "running"}

    def stop(self, *, timeout: float = 5.0) -> VideoSummary:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

        with self._lock:
            if self._stopped_at is None:
                self._stopped_at = time.time()
            if self._status == "running":
                self._status = "stopped"

            return self._build_summary()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "camera_index": self.camera_index,
                "status": self._status,
                "error": self._error,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "last_score": self._last_score,
                "completed_by_model": self._completed_by_model,
            }

    def _run(self) -> None:
        cv2 = _safe_import_cv2()

        cap = None
        try:
            self._estimator = FatigueEstimator(
                config=self.config, model_path=self.model_path
            )
            cap = _open_capture(cv2, self.camera_index, fps=int(self.config.fps))
            if cap is None or not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_index}")

            consecutive_failures = 0

            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    # If capture gets stuck, try a reopen (common on MSMF).
                    if consecutive_failures >= 60:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = _open_capture(
                            cv2, self.camera_index, fps=int(self.config.fps)
                        )
                        consecutive_failures = 0
                        if cap is None or not cap.isOpened():
                            raise RuntimeError(
                                f"Camera read failed repeatedly for index {self.camera_index}"
                            )
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                res = self._estimator.process_frame(frame)
                if isinstance(res, dict) and "fatigue_score" in res:
                    with self._lock:
                        self._last_score = float(res.get("fatigue_score"))

                if isinstance(res, dict) and res.get("analysis_complete"):
                    with self._lock:
                        self._completed_by_model = True
                        self._status = "completed"
                        self._stopped_at = time.time()
                    break

                time.sleep(0.001)

        except Exception as e:
            with self._lock:
                self._status = "failed"
                self._error = str(e)
                self._stopped_at = time.time()
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

            with self._lock:
                if self._stopped_at is None:
                    self._stopped_at = time.time()
                if self._status == "running":
                    self._status = "stopped"

    def _build_summary(self) -> VideoSummary:
        est = self._estimator
        last_score = self._last_score

        avg_score: float | None = None
        blocks: List[float] = []

        if est is not None:
            blocks = [
                float(x) for x in getattr(est, "block_results", []) if x is not None
            ]
            history = getattr(est, "history", [])
            scores = [
                float(h.get("fatigue_score"))
                for h in history
                if isinstance(h, dict) and h.get("fatigue_score") is not None
            ]
            if scores:
                avg_score = float(sum(scores) / len(scores))

        # Fallback: if we have at least one computed score but no history average
        # (e.g., very short session), expose it as the average for display.
        if avg_score is None and last_score is not None:
            avg_score = float(last_score)

        verdict: str | None = None
        if avg_score is not None:
            if avg_score > 60:
                verdict = "DANGER"
            elif avg_score > 30:
                verdict = "ATTENTION"
            else:
                verdict = "OK"

        duration: float | None = None
        if self._started_at is not None and self._stopped_at is not None:
            duration = float(self._stopped_at - self._started_at)

        return VideoSummary(
            ok=(self._status not in {"failed"}),
            status=self._status,
            camera_index=int(self.camera_index),
            started_at=self._started_at,
            stopped_at=self._stopped_at,
            duration_seconds=duration,
            fatigue_score_avg=avg_score,
            fatigue_score_last=last_score,
            fatigue_score_blocks=blocks,
            verdict=verdict,
            error=self._error,
        )
