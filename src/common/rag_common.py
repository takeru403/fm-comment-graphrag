import math
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve_top_k(
    query: str,
    documents: List[str],
    embedder: OpenAIEmbeddings,
    k: int = 8,
) -> List[Tuple[str, float]]:
    doc_vecs = embedder.embed_documents(documents)
    query_vec = embedder.embed_query(query)

    scored: List[Tuple[int, float]] = []
    for idx, vec in enumerate(doc_vecs):
        sim = cosine_similarity(query_vec, vec)
        scored.append((idx, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, k)]
    return [(documents[i], s) for i, s in top]
