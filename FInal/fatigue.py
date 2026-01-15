from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _safe_import_numpy():
    import numpy as np  # type: ignore

    return np


def _try_import_scipy_linregress():
    try:
        from scipy.stats import linregress  # type: ignore

        return linregress
    except Exception:
        return None


@dataclass
class FatigueWeights:
    # Global weights
    w_mental: float = 0.6
    w_physio: float = 0.4

    # Physio sub-weights
    sub_w_vitesse: float = 0.3
    sub_w_pause: float = 0.3
    sub_w_erreur: float = 0.2
    sub_w_latence: float = 0.2


class FatigueAnalyzer:
    """Compute a fatigue score from:

    - mental performance (grading scores)
    - typing/interaction timing (keystrokes / timeSpent)

    The front-end sends, for each answer, a dict that contains:
      - timeSpent: ms
      - keystrokes: list[{touche: str, temps: ms_since_question_start}]

    We ingest these dicts during the quiz and compute a final fatigue score
    once grading is available.
    """

    def __init__(self, weights: FatigueWeights | None = None):
        self.weights = weights or FatigueWeights()

    def _slope(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        np = _safe_import_numpy()
        x = np.arange(len(values))
        linregress = _try_import_scipy_linregress()
        if linregress is not None:
            slope, _, _, _, _ = linregress(x, values)
            return float(slope)
        # Fallback without SciPy
        return float(np.polyfit(x, np.array(values, dtype=float), 1)[0])

    def evaluate_keystrokes(self, answers_data: list[dict[str, Any]]) -> Dict[str, Any]:
        """Extract typing metrics + trends over time from answers."""

        np = _safe_import_numpy()

        session_metrics: dict[str, list[float]] = {
            "latences": [],
            "vitesses_frappe": [],  # chars/sec
            "frequences_pauses": [],
            "frequences_erreurs": [],
        }

        for entry in answers_data:
            keystrokes = entry.get("keystrokes")
            if not isinstance(keystrokes, list) or not keystrokes:
                continue

            # Latency before first key press (ms)
            try:
                start_time = float(keystrokes[0].get("temps"))
            except Exception:
                start_time = 0.0
            session_metrics["latences"].append(start_time)

            timestamps: list[float] = []
            touches: list[str] = []
            for k in keystrokes:
                if not isinstance(k, dict):
                    continue
                try:
                    timestamps.append(float(k.get("temps")))
                except Exception:
                    continue
                touches.append(str(k.get("touche")))

            if len(timestamps) < 2:
                session_metrics["vitesses_frappe"].append(0.0)
                session_metrics["frequences_pauses"].append(0.0)
                session_metrics["frequences_erreurs"].append(0.0)
                continue

            # Speed: keypresses / second
            duration_ms = float(timestamps[-1] - timestamps[0])
            cps = (len(touches) / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0
            session_metrics["vitesses_frappe"].append(float(cps))

            # Pauses > 2 seconds between keystrokes
            pauses = 0.0
            erreurs = 0.0
            in_backspace_sequence = False

            for i in range(len(timestamps)):
                if i > 0:
                    diff = float(timestamps[i] - timestamps[i - 1])
                    if diff > 2000.0:
                        pauses += 1.0

                if touches[i] == "Backspace":
                    if not in_backspace_sequence:
                        erreurs += 1.0
                        in_backspace_sequence = True
                else:
                    in_backspace_sequence = False

            session_metrics["frequences_pauses"].append(float(pauses))
            session_metrics["frequences_erreurs"].append(float(erreurs))

        means = {
            k: (float(np.mean(v)) if v else 0.0) for k, v in session_metrics.items()
        }

        return {
            "pente_vitesse": self._slope(session_metrics["vitesses_frappe"]),
            "pente_latence": self._slope(session_metrics["latences"]),
            "pente_erreurs": self._slope(session_metrics["frequences_erreurs"]),
            "pente_pauses": self._slope(session_metrics["frequences_pauses"]),
            "moyennes": means,
        }

    def evaluate(
        self, *, answers: list[dict[str, Any]], grading: dict[str, Any]
    ) -> Dict[str, Any]:
        """Return fatigue scores (0-100) + details.

        - Uses grading.average_score (0-10) as mental performance proxy.
        - Uses keystroke trends as "physio" proxy.
        """

        avg_score_llm = grading.get("average_score")
        try:
            avg_score_llm_f = (
                float(avg_score_llm) if avg_score_llm is not None else 10.0
            )
        except Exception:
            avg_score_llm_f = 10.0
        avg_score_llm_f = max(0.0, min(10.0, avg_score_llm_f))

        # Mental fatigue: invert performance
        score_fatigue_mentale = (10.0 - avg_score_llm_f) * 10.0

        frappes = self.evaluate_keystrokes(answers)

        # Normalize heuristic slopes -> 0..100
        s_vitesse = min(100.0, max(0.0, -float(frappes["pente_vitesse"]) * 500.0))
        s_latence = min(100.0, max(0.0, float(frappes["pente_latence"]) * 0.5))
        s_erreurs = min(100.0, max(0.0, float(frappes["pente_erreurs"]) * 200.0))
        s_pauses = min(100.0, max(0.0, float(frappes["pente_pauses"]) * 100.0))

        w = self.weights
        score_fatigue_physio = (
            s_vitesse * w.sub_w_vitesse
            + s_latence * w.sub_w_latence
            + s_erreurs * w.sub_w_erreur
            + s_pauses * w.sub_w_pause
        )

        score_final = (
            score_fatigue_mentale * w.w_mental + score_fatigue_physio * w.w_physio
        )

        return {
            "score_fatigue_global": round(float(score_final), 2),
            "details": {
                "fatigue_mentale": round(float(score_fatigue_mentale), 2),
                "fatigue_physio": round(float(score_fatigue_physio), 2),
                "analyse_frappe": frappes,
            },
        }


class FatigueHandler:
    """Per-test storage + fatigue computation.

    Master should call:
    - start(test_id) when quiz starts
    - ingest_answer(test_id, answer_dict) every time the UI submits an answer
    - finalize(test_id, grading, answers) when results are ready
    """

    def __init__(self, *, analyzer: FatigueAnalyzer | None = None) -> None:
        self._analyzer = analyzer or FatigueAnalyzer()
        self._answers_by_test: dict[str, list[dict[str, Any]]] = {}

    def start(self, *, test_id: str) -> Dict[str, Any]:
        tid = str(test_id)
        self._answers_by_test[tid] = []
        return {"ok": True, "status": "started"}

    def ingest_answer(self, *, test_id: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        tid = str(test_id)
        if tid not in self._answers_by_test:
            self._answers_by_test[tid] = []
        if isinstance(answer, dict):
            self._answers_by_test[tid].append(dict(answer))
        return {"ok": True}

    def finalize(
        self, *, test_id: str, grading: Dict[str, Any], answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tid = str(test_id)
        # Prefer the session's stored answers (ingested incrementally), but fall back
        # to the final answers list if needed.
        stored = self._answers_by_test.get(tid)
        use_answers = stored if isinstance(stored, list) and stored else answers
        try:
            res = self._analyzer.evaluate(answers=use_answers, grading=grading)
            return {"ok": True, "fatigue": res}
        except Exception as e:
            return {"ok": False, "error": str(e), "fatigue": {}}
