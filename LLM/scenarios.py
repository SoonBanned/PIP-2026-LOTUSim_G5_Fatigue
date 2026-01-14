from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioItem:
    id: str
    title: str
    scenario: str
    image: str | None
    reference_application: str | None


class Scenarios:
    def __init__(self, *, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._items: list[ScenarioItem] = self._load(self.csv_path)

    @staticmethod
    def _load(path: Path) -> list[ScenarioItem]:
        items: list[ScenarioItem] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("ID") or "").strip()
                title = str(row.get("Titre") or "").strip()
                scenario = str(row.get("Scenario") or "").strip()
                image = str(row.get("Image") or "").strip() or None
                reference_application = (
                    str(row.get("Application_Regle") or "").strip() or None
                )
                if not sid or not scenario:
                    continue
                items.append(
                    ScenarioItem(
                        id=sid,
                        title=title,
                        scenario=scenario,
                        image=image,
                        reference_application=reference_application,
                    )
                )
        if not items:
            raise ValueError(f"No scenario items loaded from {path}")
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
                "type": "scenario",
                "id": it.id,
                "title": it.title,
                "scenario": it.scenario,
                "image": it.image,
                "reference_application": it.reference_application,
            }
            for it in picked
        ]
