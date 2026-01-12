from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TYPE_CHECKING

import requests

try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'pypdf'. Install with: pip install pypdf"
    ) from e

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'numpy'. Install with: pip install numpy"
    ) from e

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'rank-bm25'. Install with: pip install rank-bm25"
    ) from e

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer as SentenceTransformerType  # type: ignore
else:  # pragma: no cover
    SentenceTransformerType = Any  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class _OllamaChatbot:
    """Minimal local chatbot wrapper for an OpenAI-compatible endpoint (Ollama).

    Expects Ollama's OpenAI-compatible API, typically:
      LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.base_url = (
            base_url
            or os.getenv("LOCAL_OPENAI_BASE_URL")
            or "http://localhost:11434/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("LOCAL_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
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
        text = (reasoning_text or "").strip()
        text = re.sub(r"(?is)^\s*<think>\s*|\s*</think>\s*$", "", text).strip()
        if not text:
            return ""

        marker_re = re.compile(
            r"(?im)^(final\s*answer|final|answer|response)\s*[:\-]\s*(.+?)\s*$"
        )
        matches = list(marker_re.finditer(text))
        if matches:
            return matches[-1].group(2).strip()

        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        candidate = blocks[-1] if blocks else text
        if len(candidate) > 3000:
            candidate = candidate[-3000:].strip()
        return candidate

    def reply(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: str | None = None,
        retries: int = 1,
        retry_sleep_s: float = 0.5,
    ) -> str:
        if system_prompt and not any(m.role == "system" for m in self.history):
            self.add("system", system_prompt)

        self.add("user", user_text)

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_payload(),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            payload["stop"] = [stop]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(max(1, int(retries))):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                message = (data.get("choices") or [{}])[0].get("message") or {}
                content = message.get("content")
                reasoning = message.get("reasoning")

                if isinstance(content, str) and content.strip():
                    text = content.strip()
                elif isinstance(reasoning, str) and reasoning.strip():
                    text = self._final_from_reasoning(reasoning)
                elif isinstance(content, str):
                    text = content.strip()
                else:
                    raise RuntimeError(f"Unexpected response shape: {data}")

                self.add("assistant", text)
                return text
            except Exception as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(retry_sleep_s)
                    continue
                raise

        raise RuntimeError(f"Request failed: {last_err}")

    def ping(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            resp = requests.get(url, timeout=min(10.0, self.timeout))
            return resp.status_code < 400
        except Exception:
            return False


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_STOPWORDS = {
    # EN
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "at",
    "by",
    "it",
    "this",
    "that",
    "from",
    "into",
    "over",
    "under",
    "not",
    # FR
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "et",
    "ou",
    "de",
    "du",
    "dans",
    "sur",
    "pour",
    "avec",
    "est",
    "sont",
    "été",
    "être",
    "ce",
    "cet",
    "cette",
    "ces",
    "pas",
    "plus",
}


def _tokenize(text: str) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _pdf_bytes_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()


def load_pdf_from_path(path: str | Path) -> str:
    p = Path(path)
    with open(p, "rb") as f:
        return _pdf_bytes_to_text(f.read())


_COLREG_HEAD_RE = re.compile(r"(?im)^(rule\s+\d+\b.*|annex\s+[ivx]+\b.*|appendix\b.*)$")


def _slice_tokens(text: str, *, max_tokens: int, overlap_tokens: int) -> List[str]:
    text = _normalize_ws(text)
    if not text:
        return []
    toks = text.split()
    if not toks:
        return []
    out: List[str] = []
    start = 0
    n = len(toks)
    max_tokens = max(1, int(max_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    while start < n:
        end = min(n, start + max_tokens)
        out.append(" ".join(toks[start:end]))
        if end >= n:
            break
        start = max(0, end - overlap_tokens)
    return out


def chunk_colreg(
    text: str, *, chunk_tokens: int = 500, overlap_tokens: int = 150
) -> List[Tuple[str, str]]:
    raw = (text or "").replace("\r\n", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    headings: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if not ln:
            continue
        m = _COLREG_HEAD_RE.match(ln)
        if m:
            headings.append((i, m.group(1).strip()))

    if not headings:
        joined = _normalize_ws(raw)
        return [
            ("Document", ch)
            for ch in _slice_tokens(
                joined, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens
            )
        ]

    chunks: List[Tuple[str, str]] = []
    for idx, (start_i, title) in enumerate(headings):
        end_i = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        section_text = _normalize_ws(
            " ".join([ln for ln in lines[start_i:end_i] if ln])
        )
        if not section_text:
            continue
        for ch in _slice_tokens(
            section_text, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        ):
            chunks.append((title, ch))
    return chunks


def _minmax(scores: "np.ndarray") -> "np.ndarray":
    if scores.size == 0:
        return scores
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-9:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


@dataclass
class Chunk:
    idx: int
    section: str
    text: str


@dataclass
class AttachedPDF:
    name: str
    text: str
    chunks: List[Chunk]
    bm25: BM25Okapi
    bm25_tokens: List[List[str]]
    embedder_name: str
    embeddings: Optional["np.ndarray"]


class LLM:
    """Single-file LLM wrapper that matches the evaluation notebook's RAG architecture.

    Methods:
      - __init__: loads PDF + scenario application mapping, connects to Ollama model
      - generate_quizz(n, difficulty_lists): returns list of {question, truth_answer, rule_refs, supporting_quotes}
      - grade_answer_vs_ground_truth(question, truth_answer, student_answer): returns {score_total, feedback}
      - grade_answer_vs_rule_application(scenario, student_answer, source_id/reference_application): returns {score, feedback}
    """

    # Training-consistent prompts
    SYSTEM_PROMPT_QUIZ = (
        "You generate COLREGS quiz items. "
        "Use ONLY the provided quotes as evidence; do not invent rule text."
    )
    SYSTEM_PROMPT_GRADE_TRUTH = (
        "You are a strict grader for COLREGS answers. "
        "Grade the STUDENT_ANSWER against the TRUTH_ANSWER using the RUBRIC. "
        "Do not add external facts."
    )
    SYSTEM_PROMPT_GRADE_SCENARIO = (
        "You are a strict grader for COLREGS scenario rule-application. "
        "Grade the STUDENT_ANSWER against the REFERENCE_APPLICATION text. "
        "Do not add external facts."
    )

    DEFAULT_RUBRIC = [
        {
            "id": "R1",
            "max": 5,
            "criterion": "Correctly identifies the relevant rule(s) and who must keep out of the way / stand on (if applicable).",
        },
        {
            "id": "R2",
            "max": 5,
            "criterion": "States the correct required actions/precautions (early & substantial action, avoid crossing ahead, safe speed, look-out, etc.) without adding incorrect claims.",
        },
    ]

    DEFAULT_TOPICS = [
        # Part A — General
        "Rule 1 (Application)",
        "Rule 2 (Responsibility)",
        "Rule 3 (General definitions)",
        "Rule 4 (Application of Part B)",
        # Part B — Steering and Sailing Rules (Section I)
        "Rule 5 (Look-out)",
        "Rule 6 (Safe speed)",
        "Rule 7 (Risk of collision)",
        "Rule 8 (Action to avoid collision)",
        "Rule 9 (Narrow channels)",
        "Rule 10 (Traffic separation schemes)",
        # Part B — Section II
        "Rule 11 (Application)",
        "Rule 12 (Sailing vessels)",
        "Rule 13 (Overtaking)",
        "Rule 14 (Head-on situation)",
        "Rule 15 (Crossing situation)",
        "Rule 16 (Action by give-way vessel)",
        "Rule 17 (Action by stand-on vessel)",
        "Rule 18 (Responsibilities between vessels)",
        # Part B — Section III
        "Rule 19 (Conduct of vessels in restricted visibility)",
        # Part C — Lights and Shapes
        "Rule 20 (Application - lights and shapes)",
        "Rule 21 (Definitions - lights)",
        "Rule 22 (Visibility of lights)",
        "Rule 23 (Power-driven vessels underway)",
        "Rule 24 (Towing and pushing)",
        "Rule 25 (Sailing vessels underway and vessels under oars)",
        "Rule 26 (Fishing vessels)",
        "Rule 27 (Vessels not under command or restricted in ability to manoeuvre)",
        "Rule 28 (Vessels constrained by their draught)",
        "Rule 29 (Pilot vessels)",
        "Rule 30 (Anchored vessels and vessels aground)",
        "Rule 31 (Seaplanes)",
        # Part D — Sound and Light Signals
        "Rule 32 (Definitions - sound signals)",
        "Rule 33 (Equipment for sound signals)",
        "Rule 34 (Manoeuvring and warning signals)",
        "Rule 35 (Sound signals in restricted visibility)",
        "Rule 36 (Signals to attract attention)",
        "Rule 37 (Distress signals)",
        # Part E — Exemptions
        "Rule 38 (Exemptions)",
        # Annexes (commonly tested)
        "Annex I (Positioning and technical details of lights and shapes)",
        "Annex II (Additional signals for fishing vessels)",
        "Annex III (Technical details of sound signal appliances)",
        "Annex IV (Distress signals)",
        # Cross-cutting drill topics
        "Definitions: underway vs making way vs at anchor",
        "Definitions: not under command vs restricted in ability to manoeuvre",
        "Give-way vs stand-on obligations (Rules 16-17)",
        "Collision-avoidance decision making (Rules 5-8)",
        "Lights/shapes identification (Rules 20-31)",
        "Sound signals (Rules 32-37)",
    ]

    def __init__(
        self,
        *,
        model: str = "llama3.1_fine",
        pdf_path: str | Path | None = None,
        pdf_bytes: bytes | None = None,
        pdf_text: str | None = None,
        scenario_source_json: str | Path | None = None,
        scenario_source_data: list[dict[str, Any]] | dict[str, str] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        # RAG config
        retrieve_k: int = 10,
        bm25_weight: float = 0.55,
        sem_weight: float = 0.45,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_tokens: int = 1024,
        chunk_overlap_tokens: int = 256,
        multiquery: bool = True,
        cove: bool = True,
    ) -> None:
        self.bot = _OllamaChatbot(model=model, base_url=base_url, api_key=api_key)
        self.model = model

        self.retrieve_k = int(retrieve_k)
        self.bm25_weight = float(bm25_weight)
        self.sem_weight = float(sem_weight)
        self.embedder_name = str(embedder_name)
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap_tokens = int(chunk_overlap_tokens)
        self.multiquery = bool(multiquery)
        self.cove = bool(cove)

        self._embedder: Optional[SentenceTransformerType] = None  # type: ignore
        self.attached_pdf: Optional[AttachedPDF] = None
        self.application_by_id: dict[str, str] = {}

        # PDF input (choose one). This keeps the module self-contained even if you
        # don't want to read from disk.
        if pdf_text is not None:
            self._attach_pdf_text("document", pdf_text)
        elif pdf_bytes is not None:
            self._attach_pdf_text("document", _pdf_bytes_to_text(pdf_bytes))
        elif pdf_path is not None:
            txt = load_pdf_from_path(pdf_path)
            self._attach_pdf_text(Path(pdf_path).name, txt)
        else:
            raise ValueError("Provide one of: pdf_path, pdf_bytes, pdf_text")

        if scenario_source_data is not None:
            self._load_scenario_source_data(scenario_source_data)
        elif scenario_source_json is not None:
            self._load_scenario_source(scenario_source_json)

        if not self.bot.ping():
            raise RuntimeError(
                f"Cannot reach Ollama OpenAI-compatible API at {self.bot.base_url}. "
                "Ensure Ollama is running and LOCAL_OPENAI_BASE_URL is correct."
            )

    # -------------------------
    # Internal: PDF + retrieval
    # -------------------------
    def _get_embedder(self) -> Optional[SentenceTransformerType]:
        if SentenceTransformer is None:
            return None
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def _attach_pdf_text(self, name: str, text: str) -> None:
        pairs = chunk_colreg(
            text,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )
        chunks = [
            Chunk(idx=i, section=sec, text=ch) for i, (sec, ch) in enumerate(pairs)
        ]
        bm25_tokens = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(bm25_tokens)

        embedder = self._get_embedder()
        embeddings: Optional["np.ndarray"] = None
        if embedder is not None:
            docs = [f"{c.section}. {c.text}" for c in chunks]
            emb = embedder.encode(
                docs, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings = np.asarray(emb, dtype=np.float32)

        self.attached_pdf = AttachedPDF(
            name=name,
            text=text,
            chunks=chunks,
            bm25=bm25,
            bm25_tokens=bm25_tokens,
            embedder_name=self.embedder_name,
            embeddings=embeddings,
        )

    def _hybrid_scores(
        self, query: str
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        assert self.attached_pdf is not None
        q_tokens = _tokenize(query)
        bm25_scores = np.asarray(
            self.attached_pdf.bm25.get_scores(q_tokens), dtype=float
        )
        bm25_norm = _minmax(bm25_scores)

        sem_norm = np.zeros_like(bm25_norm, dtype=float)
        if self.attached_pdf.embeddings is not None:
            embedder = self._get_embedder()
            if embedder is not None:
                q_emb = embedder.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                )
                q_emb = np.asarray(q_emb[0], dtype=np.float32)
                sem = np.dot(self.attached_pdf.embeddings, q_emb).astype(float)
                sem_norm = _minmax(sem)

        hybrid = self.bm25_weight * bm25_norm + self.sem_weight * sem_norm
        return hybrid, bm25_norm, sem_norm

    def _diverse_topk(self, indices: List[int], *, k: int) -> List[int]:
        assert self.attached_pdf is not None
        picked: List[int] = []
        seen_sections: Dict[str, int] = {}
        for idx in indices:
            sec = self.attached_pdf.chunks[idx].section
            if seen_sections.get(sec, 0) >= 2:
                continue
            picked.append(idx)
            seen_sections[sec] = seen_sections.get(sec, 0) + 1
            if len(picked) >= k:
                break
        return picked

    def _retrieve_context(self, query: str, *, k: int = 6, pool: int = 20) -> List[int]:
        if not self.attached_pdf:
            return []
        hybrid, _, _ = self._hybrid_scores(query)
        ranked = np.argsort(-hybrid)[: max(pool, k)].tolist()
        ranked = [i for i in ranked if hybrid[i] > 0]
        return self._diverse_topk(ranked, k=k)

    def _build_context(self, indices: List[int]) -> str:
        assert self.attached_pdf is not None
        if not indices:
            return ""
        by_section: Dict[str, List[int]] = {}
        section_order: List[str] = []
        for idx in indices:
            sec = self.attached_pdf.chunks[idx].section
            if sec not in by_section:
                by_section[sec] = []
                section_order.append(sec)
            by_section[sec].append(idx)

        blocks: List[str] = []
        for sec in section_order:
            blocks.append(f"=== {sec} ===")
            for idx in by_section[sec]:
                c = self.attached_pdf.chunks[idx]
                blocks.append(f"(Chunk {idx+1}) {c.text}")
            blocks.append("")
        return "\n".join(blocks).strip()

    def _generate_subqueries(self, question: str) -> List[str]:
        prompt = (
            "Generate exactly 3 short, diverse search sub-queries to retrieve passages from the COLREG document. "
            "Each sub-query should be a keyword-style query and may include Rule/Annex numbers if relevant.\n"
            "Rules:\n"
            "- Output ONLY the 3 queries, one per line\n"
            "- No explanations\n"
            "- Keep each query <= 12 words\n\n"
            f"User question: {question}"
        )
        raw = self.bot.reply(prompt, max_tokens=96, temperature=0.2)
        lines = [ln.strip(" -\t") for ln in (raw or "").splitlines() if ln.strip()]
        seen: set[str] = set()
        out: List[str] = []
        for ln in lines:
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
            if len(out) >= 3:
                break
        return out if out else [question]

    def _retrieve_context_multiquery(self, question: str, *, k: int = 6) -> List[int]:
        if not self.attached_pdf:
            return []
        init = self._retrieve_context(question, k=max(4, k), pool=30)

        try:
            queries = self._generate_subqueries(question)
        except Exception:
            # If the model is temporarily unavailable, fall back to single-query retrieval.
            return init[:k]

        merged_scores = np.zeros(len(self.attached_pdf.chunks), dtype=float)
        for q in [question] + queries:
            hybrid, _, _ = self._hybrid_scores(q)
            merged_scores = np.maximum(merged_scores, hybrid)
        ranked = np.argsort(-merged_scores)[:40].tolist()
        ranked = [i for i in ranked if merged_scores[i] > 0]
        for idx in init:
            if idx in ranked:
                ranked.remove(idx)
            ranked.insert(0, idx)
        return self._diverse_topk(ranked, k=k)

    def _cove_answer(self, question: str, *, context: str) -> str:
        draft_prompt = (
            "Answer the QUESTION using ONLY the EXCERPTS. "
            "Cite evidence like (Chunk 12). If the excerpts do not contain the answer, say you don't know.\n\n"
            f"EXCERPTS:\n{context if context else '[No relevant excerpts found]'}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Output ONLY the draft answer."
        )
        draft = self.bot.reply(draft_prompt, max_tokens=512, temperature=0.2).strip()

        ver_prompt = (
            "Create exactly 3 verification questions that would check whether the DRAFT is fully supported by the EXCERPTS. "
            "Focus on potential missing conditions, exceptions, or definitions.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"DRAFT:\n{draft}\n\n"
            "Output ONLY the 3 verification questions, one per line."
        )
        ver_raw = self.bot.reply(ver_prompt, max_tokens=128, temperature=0.2)
        ver_qs = [ln.strip(" -\t") for ln in (ver_raw or "").splitlines() if ln.strip()]
        ver_qs = ver_qs[:3] if ver_qs else []
        ver_q_block = "\n".join(f"- {q}" for q in ver_qs) if ver_qs else "- (none)"

        ver_ans_prompt = (
            "Using ONLY the EXCERPTS, answer each verification question briefly. "
            "If an answer is not supported by the EXCERPTS, write 'Not supported'.\n\n"
            f"EXCERPTS:\n{context if context else '[No relevant excerpts found]'}\n\n"
            f"VERIFICATION QUESTIONS:\n{ver_q_block}\n\n"
            "Output ONLY the answers, one per line in the same order."
        )
        ver_answers = self.bot.reply(
            ver_ans_prompt, max_tokens=256, temperature=0.2
        ).strip()

        final_prompt = (
            "You are evaluating a model. Produce the FINAL answer using ONLY the EXCERPTS. "
            "If something in the DRAFT is not supported by the verification answers, remove or correct it. "
            "Cite chunks like (Chunk 12). If the excerpts do not contain the answer, say you don't know.\n\n"
            f"EXCERPTS:\n{context if context else '[No relevant excerpts found]'}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"VERIFICATION ANSWERS:\n{ver_answers}\n\n"
            "Output ONLY the final answer."
        )
        return self.bot.reply(final_prompt, max_tokens=1024, temperature=0.2).strip()

    # -------------------------
    # Internal: prompts + JSON
    # -------------------------
    @staticmethod
    def _format_quotes(quotes: list[dict]) -> str:
        if not quotes:
            return "[No quotes provided]"
        lines = []
        for q in quotes:
            if not isinstance(q, dict):
                continue
            ref = str(q.get("ref") or "").strip()
            qt = str(q.get("quote") or "").strip()
            if not qt:
                continue
            lines.append(f"- ({ref}) {qt}" if ref else f"- {qt}")
        return "\n".join(lines) if lines else "[No quotes provided]"

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any]:
        s = (text or "").strip()
        if not s:
            raise ValueError("Empty model output")
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        raise ValueError("Could not parse JSON object from model output")

    def _quotes_for_query(
        self, query: str, *, k: int = 6, max_chars: int = 420
    ) -> list[dict]:
        if not self.attached_pdf:
            return []
        indices = (
            self._retrieve_context_multiquery(query, k=min(10, k))
            if self.multiquery
            else self._retrieve_context(query, k=min(10, k), pool=30)
        )
        out: list[dict] = []
        for idx in indices[:k]:
            c = self.attached_pdf.chunks[int(idx)]
            txt = (c.text or "").strip()
            if not txt:
                continue
            out.append({"ref": f"Chunk {int(idx) + 1}", "quote": txt[:max_chars]})
        return out

    def _call_model_json(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> dict[str, Any]:
        self.bot.reset()
        raw = self.bot.reply(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        return self._extract_first_json_object(raw)

    @staticmethod
    def _coerce_rubric(rubric: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(rubric, list) or not rubric:
            return None
        out: List[Dict[str, Any]] = []
        total = 0
        for i, it in enumerate(rubric):
            if not isinstance(it, dict):
                continue
            rid = str(it.get("id") or f"R{i+1}").strip() or f"R{i+1}"
            mx = it.get("max")
            crit = str(it.get("criterion") or "").strip()
            if not crit:
                continue
            if isinstance(mx, bool):
                mx = int(mx)
            if isinstance(mx, (int, float)) and float(mx).is_integer():
                mx_i = int(mx)
            else:
                continue
            if mx_i <= 0:
                continue
            out.append({"id": rid, "max": mx_i, "criterion": crit})
            total += mx_i

        if not out:
            return None
        if total != 10:
            return None
        return out

    def _generate_rubric_for_item(
        self, *, question: str, truth_answer: str
    ) -> List[Dict[str, Any]]:
        fix_prompt = (
            "Create a grading rubric that is coherent with the quiz QUESTION and TRUTH_ANSWER. "
            "The rubric must total exactly 10 points and have 2 to 4 criteria.\n\n"
            "Rules:\n"
            '- Output ONLY JSON: {"rubric": [{"id": str, "max": int, "criterion": str}, ...]}\n'
            "- Sum of all 'max' MUST be 10\n"
            "- Keep criteria specific to the question (not generic)\n\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"TRUTH_ANSWER:\n{truth_answer.strip()}"
        )
        obj = self._call_model_json(
            system_prompt=self.SYSTEM_PROMPT_QUIZ,
            user_prompt=fix_prompt,
            max_tokens=256,
        )
        coerced = self._coerce_rubric(obj.get("rubric"))
        return coerced or self.DEFAULT_RUBRIC

    def _load_scenario_source(self, path: str | Path) -> None:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise ValueError("scenario_source_json must be a JSON array")
        self._load_scenario_source_data(rows)

    def _load_scenario_source_data(
        self, data: list[dict[str, Any]] | dict[str, str]
    ) -> None:
        if isinstance(data, dict):
            self.application_by_id = {
                str(k).strip(): str(v).strip()
                for k, v in data.items()
                if str(k).strip() and str(v).strip()
            }
            return

        if not isinstance(data, list):
            raise ValueError(
                "scenario_source_data must be a list[dict] or dict[str,str]"
            )

        out: dict[str, str] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("ID") or "").strip()
            app = str(row.get("Application_Regle") or "").strip()
            if sid and app:
                out[sid] = app
        self.application_by_id = out

    # -------------------------
    # Public API
    # -------------------------
    def generate_quizz(
        self, n: int, difficulty_lists: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        n = max(1, int(n))
        difficulty_lists = difficulty_lists or []
        topics = [t for t in self.DEFAULT_TOPICS if str(t).strip()]
        if not topics:
            raise RuntimeError("DEFAULT_TOPICS is empty")

        # Random topic per question:
        # - sample without replacement when possible
        # - otherwise sample with replacement
        if n <= len(topics):
            picked_topics = list(np.random.choice(topics, size=n, replace=False))
        else:
            picked_topics = list(np.random.choice(topics, size=n, replace=True))

        out: List[Dict[str, Any]] = []
        for i in range(n):
            topic = str(picked_topics[i])
            difficulty = (
                str(difficulty_lists[i]).strip() if i < len(difficulty_lists) else ""
            )
            quotes = self._quotes_for_query(topic)

            user = (
                "TASK: quiz_generate\n"
                + (f"DIFFICULTY: {difficulty}\n" if difficulty else "")
                + f"RULE_REFS: {topic}\n"
                + "QUOTES:\n"
                + self._format_quotes(quotes)
                + "\n\n"
                + "Generate ONE quiz item as JSON with keys: question, truth_answer, rubric.\n"
                + "The rubric must be 2-4 criteria and sum to exactly 10 points.\n"
                + "Rubric schema: rubric = [{id: string, max: int, criterion: string}, ...]"
            )
            obj = self._call_model_json(
                system_prompt=self.SYSTEM_PROMPT_QUIZ, user_prompt=user, max_tokens=384
            )
            q = str(obj.get("question") or "").strip()
            t = str(obj.get("truth_answer") or "").strip()
            if not q or not t:
                raise RuntimeError(f"Model returned invalid quiz JSON: {obj}")

            rubric = self._coerce_rubric(obj.get("rubric"))
            if rubric is None:
                rubric = self._generate_rubric_for_item(question=q, truth_answer=t)

            out.append(
                {
                    "question": q,
                    "truth_answer": t,
                    "rubric": rubric,
                    "rule_refs": [topic],
                    "supporting_quotes": quotes,
                    "difficulty": difficulty,
                }
            )
        return out

    def grade_answer_vs_ground_truth(
        self,
        *,
        question: str,
        truth_answer: str,
        student_answer: str,
        rubric: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rubric = rubric or self.DEFAULT_RUBRIC
        support = self._quotes_for_query(f"{question}\nExpected: {truth_answer}")

        rubric_lines: list[str] = []
        for it in rubric:
            if not isinstance(it, dict):
                continue
            rid = str(it.get("id") or "").strip()
            mx = it.get("max")
            crit = str(it.get("criterion") or "").strip()
            if rid and crit and isinstance(mx, (int, float)):
                rubric_lines.append(f"- {rid} (max {int(mx)}): {crit}")
        if not rubric_lines:
            raise ValueError("Rubric is empty/invalid")

        user = (
            "TASK: grade_truth_answer\n"
            + "QUESTION:\n"
            + (question or "").strip()
            + "\n\n"
            + "TRUTH_ANSWER:\n"
            + (truth_answer or "").strip()
            + "\n\n"
            + "RUBRIC (sum to 10):\n"
            + "\n".join(rubric_lines)
            + "\n\n"
            + "SUPPORTING_QUOTES:\n"
            + self._format_quotes(support)
            + "\n\n"
            + "STUDENT_ANSWER:\n"
            + (student_answer or "").strip()
            + "\n\n"
            + 'Return ONLY JSON in the schema: {"score_total": int, "breakdown": [...], "missing_points": [...], "incorrect_claims": [...], "feedback": str}.'
        )
        obj = self._call_model_json(
            system_prompt=self.SYSTEM_PROMPT_GRADE_TRUTH,
            user_prompt=user,
            max_tokens=640,
        )
        score_total = obj.get("score_total")
        if isinstance(score_total, bool):
            score_total = int(score_total)
        if isinstance(score_total, (int, float)) and float(score_total).is_integer():
            score_total = int(score_total)
        else:
            score_total = None
        if score_total is None:
            raise RuntimeError(f"Model grading JSON missing score_total: {obj}")
        score_total = max(0, min(10, int(score_total)))
        feedback = str(obj.get("feedback") or "").strip()
        return {"score_total": score_total, "feedback": feedback, "raw": obj}

    def grade_answer_vs_rule_application(
        self,
        *,
        scenario: str,
        student_answer: str,
        source_id: str | int | None = None,
        reference_application: str | None = None,
        rule_refs: Optional[List[str]] = None,
        provide_feedback: bool = True,
    ) -> Dict[str, Any]:
        sid = str(source_id).strip() if source_id is not None else ""
        ref_app = (reference_application or "").strip()
        if not ref_app and sid:
            ref_app = (self.application_by_id.get(sid) or "").strip()
        if not ref_app:
            raise ValueError(
                "reference_application is required (or provide source_id + scenario_source_json in init)"
            )

        rr = rule_refs or []
        user = (
            "TASK: rule_application_grade\n"
            + (f"SOURCE_ID: {sid}\n" if sid else "")
            + (
                "RULE_REFS: " + ", ".join([str(x) for x in rr if str(x).strip()]) + "\n"
                if rr
                else ""
            )
            + "SCENARIO:\n"
            + (scenario or "").strip()
            + "\n\n"
            + "REFERENCE_APPLICATION:\n"
            + ref_app
            + "\n\n"
            + "STUDENT_ANSWER:\n"
            + (student_answer or "").strip()
            + "\n\n"
            + 'Return ONLY JSON: {"score": int} (0..10).'
        )
        obj = self._call_model_json(
            system_prompt=self.SYSTEM_PROMPT_GRADE_SCENARIO,
            user_prompt=user,
            max_tokens=128,
        )
        score = obj.get("score")
        if isinstance(score, bool):
            score = int(score)
        if isinstance(score, (int, float)) and float(score).is_integer():
            score = int(score)
        else:
            score = None
        if score is None:
            raise RuntimeError(f"Model scenario grading JSON missing score: {obj}")
        score = max(0, min(10, int(score)))

        feedback = ""
        if provide_feedback:
            # Keep the score call training-consistent, then do a lightweight feedback-only call.
            support = self._quotes_for_query(f"{scenario}\n{ref_app}")
            fb_prompt = (
                "You are a strict grader for COLREGS scenario rule-application. "
                "Use ONLY the REFERENCE_APPLICATION and SUPPORTING_QUOTES; do not add external facts.\n\n"
                f"SCORE: {score}/10\n\n"
                f"SCENARIO:\n{(scenario or '').strip()}\n\n"
                f"REFERENCE_APPLICATION:\n{ref_app}\n\n"
                f"SUPPORTING_QUOTES:\n{self._format_quotes(support)}\n\n"
                f"STUDENT_ANSWER:\n{(student_answer or '').strip()}\n\n"
                'Return ONLY JSON: {"feedback": str}. Keep feedback 1-3 sentences.'
            )
            fb_obj = self._call_model_json(
                system_prompt=self.SYSTEM_PROMPT_GRADE_SCENARIO,
                user_prompt=fb_prompt,
                max_tokens=192,
            )
            feedback = str(fb_obj.get("feedback") or "").strip()

        return {"score": score, "feedback": feedback, "raw": obj}
