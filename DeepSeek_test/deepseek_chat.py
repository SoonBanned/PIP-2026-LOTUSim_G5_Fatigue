from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'requests'. Install with: pip install requests"
    ) from e


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class DeepSeekChatbot:
    """Minimal chatbot wrapper for local inference via Ollama.

    Uses an OpenAI-compatible chat-completions endpoint, typically:
      LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1

    This is intentionally *local-only* (no Hugging Face Router / HF tokens).
    """

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        *,
        timeout: float | None = 120.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.base_url = (
            os.getenv("LOCAL_OPENAI_BASE_URL") or "http://localhost:11434/v1"
        )
        self.api_key = (
            os.getenv("LOCAL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        )
        self.history: list[ChatMessage] = []

    def reset(self) -> None:
        self.history.clear()

    def add(self, role: Role, content: str) -> None:
        self.history.append(ChatMessage(role=role, content=content))

    def _messages_payload(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    @staticmethod
    def _final_from_reasoning(reasoning_text: str) -> str:
        """Best-effort extraction of a final answer from a model's reasoning.

        Ollama's OpenAI-compatible endpoint may place output in `message.reasoning`
        (sometimes including chain-of-thought). This tries to return only a final
        answer-like segment.
        """

        text = reasoning_text.strip()
        # Common tags used by some models
        text = re.sub(r"(?is)^\s*<think>\s*|\s*</think>\s*$", "", text).strip()
        if not text:
            return ""

        # Prefer explicit markers if present
        marker_re = re.compile(
            r"(?im)^(final\s*answer|final|answer|response)\s*[:\-]\s*(.+?)\s*$"
        )
        matches = list(marker_re.finditer(text))
        if matches:
            return matches[-1].group(2).strip()

        # Otherwise, take the last paragraph block
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        candidate = blocks[-1] if blocks else text

        # If the "last block" still looks like meta-thinking, fall back to last sentences
        looks_like_thinking = re.compile(
            r"(?is)^(okay|alright|let\s+me|i\s+will|first,|now,|to\s+answer|the\s+user\s+wants)\b"
        )
        if looks_like_thinking.search(candidate) and len(text) > len(candidate):
            # Use the very end of the full text instead
            candidate = text[-1500:].strip()

        # Last-resort: if it's still huge, keep the tail
        if len(candidate) > 3000:
            candidate = candidate[-3000:].strip()

        return candidate

    def reply(
        self,
        user_text: str,
        *,
        system_prompt: str | None = (
            "You are a helpful assistant. Provide only the final answer. "
            "Do not include reasoning steps."
        ),
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: str | None = None,
    ) -> str:
        if system_prompt and not any(m.role == "system" for m in self.history):
            self.add("system", system_prompt)

        self.add("user", user_text)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_payload(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = [stop]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout or 120.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Local inference error {resp.status_code}: {(resp.text or '')[:300]}"
            )

        data = resp.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning")

        if isinstance(content, str) and content.strip():
            text = content.strip()
        elif isinstance(reasoning, str) and reasoning.strip():
            # Some Ollama models emit text in a non-standard `reasoning` field.
            # Extract a final answer-like segment (avoid returning chain-of-thought).
            text = self._final_from_reasoning(reasoning)
        elif isinstance(content, str):
            # content is present but empty
            text = content.strip()
        else:
            raise RuntimeError(f"Unexpected response shape: {data}")

        self.add("assistant", text)
        return text


def run_cli() -> None:
    bot = DeepSeekChatbot()
    print(f"Connected to {bot.model}. Type '/reset' or '/exit'.")

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return

        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            print("Bye")
            return
        if user.lower() == "/reset":
            bot.reset()
            print("(history cleared)")
            continue

        try:
            ans = bot.reply(user)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"Bot> {ans}\n")
