from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'requests'. Install with: pip install requests"
    ) from e

try:
    from huggingface_hub import InferenceClient
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'huggingface_hub'. Install with: pip install huggingface_hub"
    ) from e


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class DeepSeekChatbot:
    """Minimal chatbot wrapper for Hugging Face hosted inference.

    This uses the Hugging Face Inference API via huggingface_hub.InferenceClient.
    You must set HF_TOKEN in your environment.

    Notes:
    - If the model supports the chat-completions API, we use client.chat.completions.
    - Otherwise, we fall back to text-generation.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V3.2",
        *,
        token_env: str = "HF_TOKEN",
        timeout: float | None = 120.0,
    ) -> None:
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(
                f"{token_env} is not set. Create a Hugging Face access token and set it, e.g.\n"
                f'  setx {token_env} "hf_..."\n'
                f"Then restart your terminal/kernel."
            )

        self.model = model
        self.timeout = timeout

        # Preferred backend: Hugging Face Router (OpenAI-compatible)
        # The legacy Inference API endpoint (api-inference.huggingface.co) is deprecated.
        self.router_base_url = (
            os.getenv("HF_ROUTER_BASE_URL") or "https://router.huggingface.co"
        )

        # Hugging Face deprecated the legacy api-inference endpoint.
        # In huggingface_hub==0.25.x, the endpoint is hardcoded in
        # huggingface_hub.inference._client.INFERENCE_ENDPOINT.
        # Patch it to router.huggingface.co to avoid 410 Gone errors.
        try:
            import huggingface_hub.inference._client as _hf_inf_client

            desired = (
                os.getenv("HF_INFERENCE_ENDPOINT")
                or os.getenv("HF_INFERENCE_BASE_URL")
                or "https://router.huggingface.co/hf-inference"
            )
            current = getattr(_hf_inf_client, "INFERENCE_ENDPOINT", None)
            if isinstance(current, str):
                normalized = current.rstrip("/")
                desired_norm = desired.rstrip("/")
                if normalized != desired_norm and (
                    "api-inference.huggingface.co" in normalized
                    or normalized == "https://router.huggingface.co"
                ):
                    _hf_inf_client.INFERENCE_ENDPOINT = desired
        except Exception:
            pass

        self.client = InferenceClient(model=model, token=token, timeout=timeout)
        self.history: list[ChatMessage] = []
        self._token = token

    def reset(self) -> None:
        self.history.clear()

    def add(self, role: Role, content: str) -> None:
        self.history.append(ChatMessage(role=role, content=content))

    def _messages_payload(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def reply(
        self,
        user_text: str,
        *,
        system_prompt: str | None = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: str | None = None,
    ) -> str:
        if system_prompt and not any(m.role == "system" for m in self.history):
            self.add("system", system_prompt)

        self.add("user", user_text)

        # Prefer Hugging Face Router OpenAI-compatible chat endpoint.
        text: str | None = None
        router_url = f"{self.router_base_url.rstrip('/')}/v1/chat/completions"
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._messages_payload(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop:
                payload["stop"] = [stop]

            resp = requests.post(
                router_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout or 120.0,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Router error {resp.status_code}: {resp.text[:200]}"
                )

            data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                "content"
            )
            if isinstance(text, str):
                text = text.strip()
            else:
                text = None
        except Exception:
            text = None

        if text is None:
            # Fallback: try huggingface_hub inference client (may not support all models).
            try:
                resp2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages_payload(),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=[stop] if stop else None,
                )
                text = (resp2.choices[0].message.content or "").strip()
            except Exception:
                prompt = self._format_prompt_fallback(self.history)
                out = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=[stop] if stop else None,
                    return_full_text=False,
                )
                text = (out or "").strip()

        self.add("assistant", text)
        return text

    @staticmethod
    def _format_prompt_fallback(messages: Iterable[ChatMessage]) -> str:
        # Simple, model-agnostic chat formatting.
        chunks: list[str] = []
        for m in messages:
            if m.role == "system":
                chunks.append(f"System: {m.content}")
            elif m.role == "user":
                chunks.append(f"User: {m.content}")
            else:
                chunks.append(f"Assistant: {m.content}")
        chunks.append("Assistant:")
        return "\n".join(chunks)


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
