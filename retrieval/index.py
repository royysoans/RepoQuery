import json
import os
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, data_dir: str):
        manifest_path = os.path.join(data_dir, "embedding_manifest.json")
        embeddings_path = os.path.join(data_dir, "embeddings.npy")
        chunks_path = os.path.join(data_dir, "chunks.json")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

        embeddings = np.load(embeddings_path).astype("float32")

        if embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(self.chunks)} chunks. "
                "chunks.json and embeddings.npy must be built from the same run."
            )

        # IndexFlatIP = exact inner-product search. Since embeddings were
        # normalized at encode time, inner product == cosine similarity.
        # "Flat" means brute-force — perfectly fine up to ~100k-1M vectors,
        # which covers essentially any single repository. Swap for an
        # approximate index (IVF/HNSW) only if indexing a huge monorepo
        # makes search noticeably slow.
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        # Same embedding model used at build time — required, or query
        # vectors won't live in the same space as the indexed vectors.
        self.model = SentenceTransformer(self.manifest["model_name"])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer than top_k results exist
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results


def build_index_cli():
    """Quick manual sanity check: embed a query, print top matches with citations."""
    if len(sys.argv) < 2:
        print("Usage: python3 index.py <data_dir> [query]")
        sys.exit(1)

    data_dir = sys.argv[1]
    retriever = Retriever(data_dir)

    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Question: ")
    results = retriever.search(query, top_k=5)

    print(f"\nTop {len(results)} results for: {query!r}\n")
    for r in results:
        print(f"[{r['score']:.3f}] {r['file_path']}:{r['start_line']}-{r['end_line']}  "
              f"({r['symbol_type']} {r.get('symbol_name') or ''})")
        preview = r["snippet"].strip().splitlines()[0][:80]
        print(f"    {preview}")


if __name__ == "__main__":
    build_index_cli()
