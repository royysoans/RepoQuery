import json
import os
import re
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize_code(text: str) -> List[str]:
    """Tokenize code text, splitting on camelCase, snake_case, paths, and punctuation."""
    # Split camelCase: loadQuestions -> load Questions
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Split non-alphanumeric
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", s1.lower())
    tokens = []
    for t in raw_tokens:
        tokens.append(t)
        # Also split snake_case tokens
        sub_tokens = t.split("_")
        if len(sub_tokens) > 1:
            tokens.extend([st for st in sub_tokens if st])
    return tokens


class Retriever:
    def __init__(self, data_dir: str):
        manifest_path = os.path.join(data_dir, "embedding_manifest.json")
        embeddings_path = os.path.join(data_dir, "embeddings.npy")
        chunks_path = os.path.join(data_dir, "chunks.json")

        if not os.path.isfile(manifest_path) or not os.path.isfile(embeddings_path) or not os.path.isfile(chunks_path):
            raise FileNotFoundError(
                f"Missing index files in {data_dir}. Ensure chunks.json, embeddings.npy, and embedding_manifest.json exist."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        embeddings = np.load(embeddings_path).astype("float32")

        if embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(self.chunks)} chunks. "
                "chunks.json and embeddings.npy must be built from the same run."
            )
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.model = SentenceTransformer(self.manifest["model_name"])

        # Initialize BM25 Sparse Index
        self._init_bm25()

    def _init_bm25(self):
        if not BM25Okapi:
            self.bm25 = None
            return

        corpus_tokens = []
        for c in self.chunks:
            # Index file path, symbol name, and snippet
            text = f"{c['file_path']} {c.get('symbol_name') or ''} {c['snippet']}"
            corpus_tokens.append(tokenize_code(text))

        self.bm25 = BM25Okapi(corpus_tokens)

    def search_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Vector semantic search via FAISS."""
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["dense_score"] = float(score)
            chunk["chunk_index"] = int(idx)
            results.append(chunk)

        return results

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Keyword sparse search via BM25."""
        if not self.bm25:
            return []

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = score
            chunk["chunk_index"] = int(idx)
            results.append(chunk)

        return results

    def search(self, query: str, top_k: int = 5, dense_weight: float = 0.6, bm25_weight: float = 0.4) -> List[Dict[str, Any]]:
        """
        Hybrid search combining Dense FAISS and BM25 using Reciprocal Rank Fusion (RRF).
        """
        candidate_k = max(top_k * 3, 20)
        dense_results = self.search_dense(query, top_k=candidate_k)
        bm25_results = self.search_bm25(query, top_k=candidate_k)

        if not bm25_results:
            # Fallback to pure dense if BM25 is unavailable or returned no hits
            for r in dense_results:
                r["score"] = r.get("dense_score", 0.0)
            return dense_results[:top_k]

        # Reciprocal Rank Fusion (RRF) with constant k=60
        RRF_K = 60.0
        scores_by_idx: Dict[int, float] = {}
        chunks_by_idx: Dict[int, dict] = {}

        for rank, item in enumerate(dense_results):
            idx = item["chunk_index"]
            scores_by_idx[idx] = scores_by_idx.get(idx, 0.0) + (dense_weight / (RRF_K + rank + 1))
            chunks_by_idx[idx] = item

        for rank, item in enumerate(bm25_results):
            idx = item["chunk_index"]
            scores_by_idx[idx] = scores_by_idx.get(idx, 0.0) + (bm25_weight / (RRF_K + rank + 1))
            if idx not in chunks_by_idx:
                chunks_by_idx[idx] = item
            else:
                chunks_by_idx[idx]["bm25_score"] = item.get("bm25_score")

        sorted_indices = sorted(scores_by_idx.keys(), key=lambda i: scores_by_idx[i], reverse=True)

        final_results = []
        for idx in sorted_indices[:top_k]:
            chunk = dict(chunks_by_idx[idx])
            chunk["score"] = float(scores_by_idx[idx])
            final_results.append(chunk)

        return final_results


def build_index_cli():
    """CLI test: search a query with hybrid search, print top matches with citations."""
    if len(sys.argv) < 2:
        print("Usage: python3 index.py <data_dir> [query]")
        sys.exit(1)

    data_dir = sys.argv[1]
    retriever = Retriever(data_dir)

    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Question: ")
    results = retriever.search(query, top_k=5)

    print(f"\nTop {len(results)} hybrid search results for: {query!r}\n")
    for r in results:
        sym = f" ({r['symbol_type']} {r.get('symbol_name') or ''})"
        print(f"[{r['score']:.4f}] {r['file_path']}:{r['start_line']}-{r['end_line']}{sym}")
        preview = r["snippet"].strip().splitlines()[0][:80]
        print(f"    {preview}")


if __name__ == "__main__":
    build_index_cli()

