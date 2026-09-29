# Pipeline: Saare 4 steps (Embed -> FAISS Search -> Grounded Prompt -> LLM Answer) ko ek sath chalana
from typing import List, Dict, Any, Optional
import faiss
import openai

from .embeddings import embed
from .vector_store import search_index
from .generator import build_grounded_prompt, generate_answer


# End-to-end RAG query orchestrator
def run_rag_query(
    user_question: str,
    chunks: List[str],
    index: faiss.IndexFlatL2,
    client: openai.OpenAI,
    embedding_model: str = "openai/text-embedding-3-small",
    chat_model: str = "openai/gpt-4o-mini",
    top_k: int = 3,
    custom_system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    # 1. Question ka vector banao
    q_vec = embed([user_question], client=client, model=embedding_model)[0]

    # 2. FAISS se top_k closest chunks dhoondo
    distances, indices = search_index(index, q_vec, top_k=top_k)

    # 3. Chunks extract karo
    matched_sources = [{"rank": i + 1, "dist": float(d), "text": chunks[idx]} for i, (d, idx) in enumerate(zip(distances, indices)) if 0 <= idx < len(chunks)]

    # 4. Grounded prompt banao aur LLM se answer lo
    messages = build_grounded_prompt(user_question, [s["text"] for s in matched_sources], custom_system_prompt)
    answer = generate_answer(messages, client=client, model=chat_model)

    return {
        "answer": answer,
        "sources": matched_sources,
        "context": "\n\n".join([f"[Source {s['rank']}]\n{s['text']}" for s in matched_sources])
    }
