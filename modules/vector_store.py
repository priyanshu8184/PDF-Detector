# Vector Store: FAISS se lakho vectors me se millisecond me closest matches dhoondna
from typing import Tuple
import numpy as np
import faiss


# FAISS Flat L2 index banana aur vectors add karna
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings = np.array(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings 2D non-empty array honi chahiye.")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# User ke question ke vector se top_k nearest chunks nikalna (Kam distance = Better match)
def search_index(index: faiss.IndexFlatL2, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    top_k = min(top_k, index.ntotal)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)
    return distances[0], indices[0]
