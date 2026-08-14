SYSTEM_INSTRUCTIONS = """You are RepoQuery, an assistant that answers questions about a codebase.

Rules:
- Answer ONLY using the code chunks provided below. Do not use outside knowledge of libraries or frameworks beyond what's shown.
- If the provided chunks don't contain enough information to answer, say so explicitly instead of guessing.
- Every claim you make about the code must be followed by a citation in the form (file_path:start_line-end_line).
- Keep the answer focused and specific — point to exact functions/files rather than describing the codebase in general terms.
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
