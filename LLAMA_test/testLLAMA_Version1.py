!pip -q uninstall -y transformers sentence-transformers huggingface_hub tokenizers accelerate
!pip -q install -U \
"huggingface_hub<1.0,>=0.34.0" \
"transformers==4.46.3" \
"tokenizers==0.20.3" \
"sentence-transformers==3.2.1" \
"accelerate>=0.33.0" \
"pypdf" "faiss-cpu"

# =========================
# 0) Install (Colab)
# =========================
!pip -q install -U transformers accelerate sentence-transformers pypdf faiss-cpu

import re, math
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# =========================
# 1) Upload PDF (Colab)
# =========================
from google.colab import files
uploaded = files.upload()  # choisis ton PDF
pdf_path = next(iter(uploaded.keys()))
print("PDF:", pdf_path)

# =========================
# 2) PDF -> texte
# =========================
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        # nettoyage léger
        txt = txt.replace("\u00ad", "")  # soft hyphen
        txt = re.sub(r"[ \t]+", " ", txt)
        parts.append(txt.strip())
    return "\n\n".join([p for p in parts if p])

raw_text = pdf_to_text(pdf_path)
print("Chars:", len(raw_text))

# =========================
# 3) Chunking (découpage)
# =========================
def chunk_text(text: str, chunk_size=900, overlap=150):
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

chunks = chunk_text(raw_text, chunk_size=900, overlap=150)
print("Nb chunks:", len(chunks))
print("Ex chunk:\n", chunks[0][:400], "...")

# =========================
# 4) Embeddings + FAISS index
# =========================
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL)

# encode en float32 (FAISS)
emb = embedder.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb = np.array(emb, dtype="float32")

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner Product (ok si normalize_embeddings=True)
index.add(emb)

def retrieve(query: str, k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append((int(idx), float(score), chunks[int(idx)]))
    return results

!pip -q install -U "huggingface_hub<1.0,>=0.34.0"

from huggingface_hub import login
login()

# =========================
# 5) Charger LLaMA (ou autre modèle HF)
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def format_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    system = (
        "You are an assistant who only responds based on the provided CONTEXT."
    )

    # Si le tokenizer a un chat template (instruct models), on l'utilise
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXTE:\n{context_block}\n\nQUESTION:\n{question}"}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # fallback "prompt texte"
    return (
        f"SYSTEM:\n{system}\n\n"
        f"CONTEXTE:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RÉPONSE:\n"
    )

def ask_pdf(question: str, k=5, max_new_tokens=300, temperature=0.2):
    hits = retrieve(question, k=k)
    contexts = [f"[chunk {idx} | score={score:.3f}]\n{text}" for idx, score, text in hits]

    prompt = format_prompt(question, contexts)

    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        return_full_text=False
    )[0]["generated_text"]

    return out, hits

# =========================
# 6) Exemple d'utilisation
# =========================
question = "A sailing vessel is overtaking a power-driven vessel. According to the interaction between Rule 13 and Rule 18, which vessel is the \"stand-on\" vessel and why?"
answer, sources = ask_pdf(question, k=5)
print("RÉPONSE:\n", answer)
