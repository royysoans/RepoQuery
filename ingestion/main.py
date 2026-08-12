import json
import os
import sys
import dataclasses

from scanner import scan_repository
from chunker import chunk_file


def build_chunks_json(repo_root: str, output_path: str) -> int:
    records = scan_repository(repo_root)

    all_chunks = []
    for record in records:
        for chunk in chunk_file(record):
            all_chunks.append(dataclasses.asdict(chunk))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return len(all_chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <repo_path> [output_path]")
        sys.exit(1)

    repo_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chunks.json"
    )

    count = build_chunks_json(repo_path, out_path)
    print(f"Wrote {count} chunks to {out_path}")