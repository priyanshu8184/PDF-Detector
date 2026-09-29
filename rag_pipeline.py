# Backward-compatible wrapper
from modules import (
    fixed_size_chunks, semantic_chunks, recursive_chunks,
    get_openai_client, embed, cosine_similarity,
    build_faiss_index, search_index,
    build_grounded_prompt, generate_answer,
    run_rag_query
)

rag_query = run_rag_query

__all__ = [
    "fixed_size_chunks", "semantic_chunks", "recursive_chunks",
    "get_openai_client", "embed", "cosine_similarity",
    "build_faiss_index", "search_index",
    "build_grounded_prompt", "generate_answer",
    "run_rag_query", "rag_query"
]
