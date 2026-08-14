#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingestion.main import build_chunks_json
from embeddings.embedder import embed_chunks
from retrieval.index import Retriever
from llm.answer import ask_question


def get_default_data_dir(repo_path: str) -> str:
    repo_name = os.path.basename(os.path.abspath(repo_path))
    path_hash = hashlib.md5(os.path.abspath(repo_path).encode("utf-8")).hexdigest()[:6]
    return os.path.join(PROJECT_ROOT, "data", f"{repo_name}_{path_hash}")


def is_index_ready(data_dir: str) -> bool:
    manifest = os.path.join(data_dir, "embedding_manifest.json")
    embeddings = os.path.join(data_dir, "embeddings.npy")
    chunks = os.path.join(data_dir, "chunks.json")
    return os.path.isfile(manifest) and os.path.isfile(embeddings) and os.path.isfile(chunks)


def index_repo(repo_path: str, data_dir: str, force: bool = False) -> str:
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise ValueError(f"Target repository path is not a valid directory: {repo_path}")

    os.makedirs(data_dir, exist_ok=True)
    chunks_path = os.path.join(data_dir, "chunks.json")

    if not force and is_index_ready(data_dir):
        print(f"Using existing index at: {data_dir}")
        return data_dir

    print(f"\n[1/2] Scanning & chunking '{repo_path}'...")
    chunk_count = build_chunks_json(repo_path, chunks_path)
    print(f"Generated {chunk_count} chunks.")

    print(f"[2/2] Embedding chunks into vector index...")
    embed_chunks(chunks_path, data_dir)
    print(f"Index successfully built at {data_dir}\n")

    return data_dir


def cmd_index(args: argparse.Namespace):
    data_dir = args.data_dir or get_default_data_dir(args.repo)
    index_repo(args.repo, data_dir, force=True)


def cmd_search(args: argparse.Namespace):
    data_dir = args.data_dir or get_default_data_dir(args.repo)
    index_repo(args.repo, data_dir, force=args.reindex)

    retriever = Retriever(data_dir)
    query = args.query

    if not query:
        print("Starting interactive search (Ctrl+C to exit):")
        while True:
            try:
                query = input("\nSearch Query > ").strip()
                if not query:
                    continue
                _run_search(retriever, query, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting search.")
                break
    else:
        _run_search(retriever, query, top_k=args.top_k)


def _run_search(retriever: Retriever, query: str, top_k: int = 5):
    results = retriever.search(query, top_k=top_k)
    print(f"\n--- Top {len(results)} Matches for: {query!r} ---")
    for r in results:
        sym = f" ({r['symbol_type']} {r.get('symbol_name') or ''})"
        print(f"\n[{r['score']:.4f}] {r['file_path']}:{r['start_line']}-{r['end_line']}{sym}")
        preview = "\n".join(r["snippet"].strip().splitlines()[:6])
        print(f"    {preview}")


def cmd_ask(args: argparse.Namespace):
    data_dir = args.data_dir or get_default_data_dir(args.repo)
    index_repo(args.repo, data_dir, force=args.reindex)

    retriever = Retriever(data_dir)
    question = args.question

    if not question:
        print("Starting interactive Q&A session (Ctrl+C to exit):")
        while True:
            try:
                question = input("\nQuestion > ").strip()
                if not question:
                    continue
                _run_ask(retriever, question, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
    else:
        _run_ask(retriever, question, top_k=args.top_k)


def _run_ask(retriever: Retriever, question: str, top_k: int = 5):
    print(f"\nSearching & generating answer...")
    result = ask_question(retriever, question, top_k=top_k)

    print(f"\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print("-" * 60)
    print(f"A: {result['answer']}")
    print("-" * 60)
    print("Sources & Citations:")
    for c in result["citations"]:
        symbol = f" ({c['symbol_name']})" if c.get("symbol_name") else ""
        print(f"  [{c['score']:.4f}] {c['file_path']}:{c['start_line']}-{c['end_line']}{symbol}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="repoquery",
        description="RepoQuery: Semantic & Hybrid RAG Codebase Search and Question Answering"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    parser_ask = subparsers.add_parser("ask", help="Ask a question about a repository codebase")
    parser_ask.add_argument("repo", help="Path to repository")
    parser_ask.add_argument("question", nargs="?", default=None, help="Question to ask (interactive if omitted)")
    parser_ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser_ask.add_argument("--data-dir", default=None, help="Custom data directory for index")
    parser_ask.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser_ask.set_defaults(func=cmd_ask)

    parser_search = subparsers.add_parser("search", help="Perform hybrid search without LLM generation")
    parser_search.add_argument("repo", help="Path to repository")
    parser_search.add_argument("query", nargs="?", default=None, help="Query string (interactive if omitted)")
    parser_search.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser_search.add_argument("--data-dir", default=None, help="Custom data directory for index")
    parser_search.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser_search.set_defaults(func=cmd_search)

    parser_index = subparsers.add_parser("index", help="Explicitly build index for a repository")
    parser_index.add_argument("repo", help="Path to repository")
    parser_index.add_argument("--data-dir", default=None, help="Custom data directory for index")
    parser_index.set_defaults(func=cmd_index)

    if len(sys.argv) > 1 and sys.argv[1] not in ("ask", "search", "index", "-h", "--help"):
        args = parser.parse_args(["ask"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
