from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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
        print("Starting the LLM")
        LLM = _import_llm_class()
        try:
            self.llm = LLM(model=self.config.llm_model, **(llm_kwargs or {}))
            print("LLM initialized")
        except Exception as e:
            self.llm = None
            self.llm_error = f"LLM init failed: {e}"
            print(self.llm_error)

        Interface = _import_interface_class()
        self.interface = Interface(
            master=self, host=self.config.host, port=self.config.port
        )

        self._quiz_lock = asyncio.Lock()
        self._quiz_items: list[dict[str, Any]] = []
        self._quiz_meta: dict[str, Any] = {}

    # -------------------------
    # Quiz generation / serving
    # -------------------------
    def reset_quiz(self) -> None:
        self._quiz_items = []
        self._quiz_meta = {}

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

    async def _ensure_quiz(self, *, n: int = 4, difficulty: str = "easy") -> None:
        n = int(n)
        difficulty = str(difficulty or "easy")
        async with self._quiz_lock:
            if (
                self._quiz_items
                and self._quiz_meta.get("n") == n
                and self._quiz_meta.get("difficulty") == difficulty
            ):
                return

            self._quiz_meta = {"n": n, "difficulty": difficulty}

            if self.llm is None:
                self._quiz_items = self._fallback_quiz(n)
                return

            difficulty_lists = [difficulty] * n
            last_err: Exception | None = None
            for attempt in range(1, 4):
                try:
                    items = await asyncio.to_thread(
                        self.llm.generate_quizz, n, difficulty_lists
                    )
                    if not isinstance(items, list) or not items:
                        raise RuntimeError("LLM returned empty quiz")
                    self._quiz_items = items
                    self.llm_error = ""
                    return
                except Exception as e:
                    last_err = e
                    self.llm_error = f"Quiz generation failed: {e}"
                    print(self.llm_error)
                    if attempt < 3:
                        print("Retrying quiz generation")
                        await asyncio.sleep(0.2)
                        continue

            # Keep the UI usable even if generation fails repeatedly.
            self._quiz_items = self._fallback_quiz(n)
            if last_err is not None:
                self.llm_error = f"Quiz generation failed (fallback used): {last_err}"

    async def start_test(
        self, *, n: int = 4, difficulty: str = "easy"
    ) -> Dict[str, Any]:
        self.reset_quiz()
        await self._ensure_quiz(n=n, difficulty=difficulty)
        print(self._quiz_meta)
        return {
            "ok": True,
            "total": len(self._quiz_items),
            "meta": self._quiz_meta,
            "llm_ready": self.llm is not None,
            "llm_error": self.llm_error,
        }

    async def get_question(self, *, index: int) -> Dict[str, Any]:
        await self._ensure_quiz(
            n=int(self._quiz_meta.get("n", 4)),
            difficulty=str(self._quiz_meta.get("difficulty", "easy")),
        )
        i = int(index)
        total = len(self._quiz_items)
        if i < 0 or i >= total:
            return {"done": True, "index": i, "total": total}

        item = self._quiz_items[i]
        question = (item.get("question") if isinstance(item, dict) else None) or str(
            item
        )
        return {
            "done": False,
            "index": i,
            "total": total,
            "question": question,
            "image": None,
        }

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
    await master.run()


if __name__ == "__main__":
    asyncio.run(main())
