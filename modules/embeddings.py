# Embeddings: Text ko 1536-dimensional numbers (Vectors) me badalna aur angle naapna
import os
from typing import List, Optional
import numpy as np
from numpy.linalg import norm
import openai


# OpenAI / OpenRouter client initialize karna
def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> openai.OpenAI:
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("API key missing hai! Please sidebar ya .env me API key enter karein.")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


# Text chunks ko float vectors me badalna (Batching se rate-limit nahi lagti)
def embed(texts: List[str], client: openai.OpenAI, model: str = "openai/text-embedding-3-small", batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in res.data])
    return np.array(vectors, dtype="float32")


# Cosine Similarity: Do vectors ka angle naapna (1.0 = same meaning, 0.0 = unrelated)
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = norm(v1), norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
