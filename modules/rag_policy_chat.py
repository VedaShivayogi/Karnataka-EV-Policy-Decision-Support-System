"""
Policy Chat (RAG)
Answers questions like: "What happens if Karnataka increases subsidies by ₹5,000?"
by retrieving relevant chunks from official policy documents (data/policy_docs/*.txt)
and grounding the LLM's answer in them.

Run:
    python modules/rag_policy_chat.py "What happens if Karnataka increases subsidies by 5000?"
"""

import sys
import os
import glob
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import chat

INDEX_FILE = os.path.join(config.VECTOR_DB_PATH, "faiss.index")
META_FILE = os.path.join(config.VECTOR_DB_PATH, "meta.pkl")


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def build_index(policy_dir=os.path.join(config.DATA_DIR, "policy_docs")):
    import faiss

    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
    embedder = _get_embedder()

    docs, metas = [], []
    for path in glob.glob(os.path.join(policy_dir, "*.txt")):
        with open(path, "r") as f:
            text = f.read()
        for chunk in chunk_text(text):
            docs.append(chunk)
            metas.append({"source": os.path.basename(path)})

    if not docs:
        raise RuntimeError(f"No .txt policy documents found in {policy_dir}")

    embeddings = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"docs": docs, "metas": metas}, f)

    print(f"Indexed {len(docs)} chunks from {policy_dir} -> {INDEX_FILE}")


def retrieve(query, k=4):
    import faiss

    if not os.path.exists(INDEX_FILE):
        build_index()

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        store = pickle.load(f)

    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype="float32"), k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append(
            {"text": store["docs"][idx], "source": store["metas"][idx]["source"], "score": float(score)}
        )
    return results


SYSTEM_PROMPT = (
    "You are a Karnataka EV policy assistant. Answer ONLY using the provided context "
    "from official policy documents. If the answer requires numeric extrapolation "
    "(e.g. subsidy increases), reason step by step using any figures found in the "
    "context, and clearly state your assumptions. If the context doesn't cover the "
    "question, say so honestly instead of guessing."
)


def ask(question, k=4):
    chunks = retrieve(question, k=k)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer = chat(prompt, system=SYSTEM_PROMPT)
    return answer, chunks


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What happens if Karnataka increases subsidies by ₹5,000?"
    answer, sources = ask(q)
    print(f"Q: {q}\n\nA: {answer}\n")
    print("Sources used:")
    for s in sources:
        print(f"  - {s['source']} (score={s['score']:.3f})")
