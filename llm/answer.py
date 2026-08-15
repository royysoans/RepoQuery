
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from retrieval.index import Retriever
    from .prompt import build_prompt
    from .gemini_client import generate_answer, generate_answer_stream
except ImportError:
    from retrieval.index import Retriever
    from prompt import build_prompt
    from gemini_client import generate_answer, generate_answer_stream



DEFAULT_TOP_K = 5


def ask_question(retriever: Retriever, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    chunks = retriever.search(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant code was found in the index for this question.",
            "citations": [],
        }

    prompt = build_prompt(question, chunks)
    answer_text = generate_answer(prompt)

    citations = [
        {
            "file_path": c["file_path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "symbol_name": c.get("symbol_name"),
            "score": c["score"],
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
    }


def ask_question_stream(retriever: Retriever, question: str, top_k: int = DEFAULT_TOP_K):
    """
    Search context and return (stream_generator, citations).
    """
    chunks = retriever.search(question, top_k=top_k)
    citations = [
        {
            "file_path": c["file_path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "symbol_name": c.get("symbol_name"),
            "score": c["score"],
        }
        for c in chunks
    ]

    if not chunks:
        def empty_gen():
            yield "No relevant code was found in the index for this question."
        return empty_gen(), citations

    prompt = build_prompt(question, chunks)
    return generate_answer_stream(prompt), citations



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 answer.py <data_dir> [question]")
        sys.exit(1)

    data_dir = sys.argv[1]
    retriever = Retriever(data_dir)

    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Question: ")
    result = ask_question(retriever, question)

    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for c in result["citations"]:
        symbol = f" ({c['symbol_name']})" if c["symbol_name"] else ""
        print(f"  [{c['score']:.3f}] {c['file_path']}:{c['start_line']}-{c['end_line']}{symbol}")
