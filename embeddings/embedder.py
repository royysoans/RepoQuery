import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def build_embedding_text(chunk: dict) -> str:
    header_bits = [chunk["file_path"]]
    if chunk.get("symbol_name"):
        header_bits.append(f"{chunk['symbol_type']} {chunk['symbol_name']}")
    header = " | ".join(header_bits)
    return f"{header}\n{chunk['snippet']}"


def embed_chunks(chunks_path: str, output_dir: str, model_name: str = DEFAULT_MODEL_NAME) -> None:
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path} — run ingestion first.")

    texts = [build_embedding_text(c) for c in chunks]

    print(f"Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so FAISS inner-product == cosine similarity
    ).astype("float32")

    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)

    manifest_path = os.path.join(output_dir, "embedding_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "num_chunks": len(chunks),
            "dim": int(embeddings.shape[1]),
            "chunks_source": os.path.abspath(chunks_path),
        }, f, indent=2)

    print(f"Saved {embeddings.shape} embeddings to {embeddings_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 embedder.py <chunks.json path> [output_dir]")
        sys.exit(1)

    chunks_json_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    embed_chunks(chunks_json_path, out_dir)
