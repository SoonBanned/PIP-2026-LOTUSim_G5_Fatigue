from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QcmItem:
    id: int | str
    question: str
    choices: list[str]
    correct_index: int  # 0-based
    difficulty: str | None = None


class QCM:
    def __init__(self, *, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self._items: list[QcmItem] = self._load(self.json_path)

    @staticmethod
    def _load(path: Path) -> list[QcmItem]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("QCM JSON must be a list")

        items: list[QcmItem] = []
        for obj in raw:
            if not isinstance(obj, dict):
                continue
            qid = obj.get("id")
            question = str(obj.get("question") or "").strip()
            choices = obj.get("choices")
            correct_answer = obj.get("correct_answer")
            difficulty = obj.get("difficulty")

            if not question or not isinstance(choices, list) or len(choices) < 2:
                continue
            try:
                # Dataset is 1-based.
                correct_index = int(correct_answer) - 1
            except Exception:
                continue
            if correct_index < 0 or correct_index >= len(choices):
                continue

            items.append(
                QcmItem(
                    id=qid,
                    question=question,
                    choices=[str(c) for c in choices],
                    correct_index=correct_index,
                    difficulty=str(difficulty) if difficulty is not None else None,
                )
            )
        if not items:
            raise ValueError(f"No valid QCM items loaded from {path}")
        return items

    def sample(self, *, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        n = max(1, int(n))
        rng = random.Random(seed)
        population = self._items
        if n >= len(population):
            picked = list(population)
            rng.shuffle(picked)
        else:
            picked = rng.sample(population, n)

        return [
            {
                "type": "qcm",
                "id": it.id,
                "question": it.question,
                "choices": list(it.choices),
                # keep correct_index in session (backend only)
                "correct_index": it.correct_index,
            }
            for it in picked
        ]
