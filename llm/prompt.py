SYSTEM_INSTRUCTIONS = """
You are RepoQuery.

Rules:
- Answer ONLY from the provided chunks.
- Do not use outside knowledge.
- If information is missing, say so.
- Every factual statement MUST include a citation.
- Citation format:
  (file_path:start_line-end_line)
"""


def format_chunk(chunk: dict, index: int) -> str:
    citation = f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
    symbol = f" ({chunk['symbol_type']} {chunk['symbol_name']})" if chunk.get("symbol_name") else ""
    return (
        f"--- Chunk {index} | {citation}{symbol} ---\n"
        f"{chunk['snippet']}\n"
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    chunks_text = "\n".join(format_chunk(c, i + 1) for i, c in enumerate(chunks))

    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"Retrieved code chunks:\n\n"
        f"{chunks_text}\n"
        f"Question: {question}\n\n"
        f"Answer (with citations):"
    )
