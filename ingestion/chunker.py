import ast
import re
from dataclasses import dataclass
from typing import List, Optional

from scanner import FileRecord


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: Optional[str]
    symbol_type: str
    snippet: str


def chunk_file(record: FileRecord) -> List[Chunk]:
    if record.language == "python":
        chunks = _chunk_python(record)
    elif record.language in ("javascript", "typescript", "jsx", "tsx"):
        chunks = _chunk_js_ts(record)
    elif record.language == "markdown":
        chunks = _chunk_markdown(record)
    else:
        chunks = []

    if not chunks:
        chunks = [_whole_file_chunk(record)]

    return chunks


def _chunk_python(record: FileRecord) -> List[Chunk]:
    try:
        tree = ast.parse(record.content, filename=record.file_path)
    except SyntaxError:
        return []

    lines = record.content.splitlines()
    chunks: List[Chunk] = []
    pending_module_lines: List[int] = []

    def flush_module_group():
        if not pending_module_lines:
            return
        start = pending_module_lines[0]
        end = pending_module_lines[-1]
        snippet = "\n".join(lines[start - 1:end])
        if snippet.strip():
            chunks.append(Chunk(
                chunk_id=f"{record.file_path}:{start}-{end}",
                file_path=record.file_path,
                language=record.language,
                start_line=start,
                end_line=end,
                symbol_name=None,
                symbol_type="module",
                snippet=snippet,
            ))
        pending_module_lines.clear()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush_module_group()

            start = node.lineno
            end = _get_end_line(node, lines)
            snippet = "\n".join(lines[start - 1:end])
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"

            chunks.append(Chunk(
                chunk_id=f"{record.file_path}:{start}-{end}",
                file_path=record.file_path,
                language=record.language,
                start_line=start,
                end_line=end,
                symbol_name=node.name,
                symbol_type=symbol_type,
                snippet=snippet,
            ))
        else:
            start = node.lineno
            end = _get_end_line(node, lines)
            pending_module_lines.extend(range(start, end + 1))

    flush_module_group()

    chunks.sort(key=lambda c: c.start_line)
    return chunks


def _get_end_line(node: ast.AST, lines: List[str]) -> int:
    end = getattr(node, "end_lineno", None)
    return end if end is not None else len(lines)


_JS_BOUNDARY_RE = re.compile(
    r"^\s*(export\s+)?(default\s+)?"
    r"(async\s+)?"
    r"(function\s+(?P<func_name>\w+)|"
    r"class\s+(?P<class_name>\w+)|"
    r"const\s+(?P<const_name>\w+)\s*=\s*(async\s+)?(\([^)]*\)|\w+)\s*=>)",
)


def _chunk_js_ts(record: FileRecord) -> List[Chunk]:
    lines = record.content.splitlines()
    boundaries = []

    for i, line in enumerate(lines):
        m = _JS_BOUNDARY_RE.match(line)
        if m:
            name = m.group("func_name") or m.group("class_name") or m.group("const_name")
            symbol_type = "class" if m.group("class_name") else "function"
            boundaries.append((i, name, symbol_type))

    if not boundaries:
        return []

    chunks: List[Chunk] = []

    first_boundary_line = boundaries[0][0]
    if first_boundary_line > 0:
        preamble = "\n".join(lines[0:first_boundary_line])
        if preamble.strip():
            chunks.append(Chunk(
                chunk_id=f"{record.file_path}:1-{first_boundary_line}",
                file_path=record.file_path,
                language=record.language,
                start_line=1,
                end_line=first_boundary_line,
                symbol_name=None,
                symbol_type="module",
                snippet=preamble,
            ))

    for idx, (line_idx, name, symbol_type) in enumerate(boundaries):
        start = line_idx + 1
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        snippet = "\n".join(lines[start - 1:end])

        chunks.append(Chunk(
            chunk_id=f"{record.file_path}:{start}-{end}",
            file_path=record.file_path,
            language=record.language,
            start_line=start,
            end_line=end,
            symbol_name=name,
            symbol_type=symbol_type,
            snippet=snippet,
        ))

    return chunks


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _chunk_markdown(record: FileRecord) -> List[Chunk]:
    lines = record.content.splitlines()
    headings = []

    for i, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line)
        if m:
            headings.append((i, m.group(2).strip()))

    if not headings:
        return []

    chunks: List[Chunk] = []

    first_heading_line = headings[0][0]
    if first_heading_line > 0:
        preamble = "\n".join(lines[0:first_heading_line])
        if preamble.strip():
            chunks.append(Chunk(
                chunk_id=f"{record.file_path}:1-{first_heading_line}",
                file_path=record.file_path,
                language=record.language,
                start_line=1,
                end_line=first_heading_line,
                symbol_name=None,
                symbol_type="module",
                snippet=preamble,
            ))

    for idx, (line_idx, title) in enumerate(headings):
        start = line_idx + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        snippet = "\n".join(lines[start - 1:end])

        chunks.append(Chunk(
            chunk_id=f"{record.file_path}:{start}-{end}",
            file_path=record.file_path,
            language=record.language,
            start_line=start,
            end_line=end,
            symbol_name=title,
            symbol_type="markdown_section",
            snippet=snippet,
        ))

    return chunks


def _whole_file_chunk(record: FileRecord) -> Chunk:
    lines = record.content.splitlines()
    return Chunk(
        chunk_id=f"{record.file_path}:1-{len(lines)}",
        file_path=record.file_path,
        language=record.language,
        start_line=1,
        end_line=len(lines) if lines else 1,
        symbol_name=None,
        symbol_type="module",
        snippet=record.content,
    )


if __name__ == "__main__":
    import sys
    from scanner import scan_repository

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    records = scan_repository(target)

    all_chunks: List[Chunk] = []
    for r in records:
        all_chunks.extend(chunk_file(r))

    print(f"Chunked {len(records)} files into {len(all_chunks)} chunks\n")
    for c in all_chunks[:15]:
        preview = c.snippet.strip().splitlines()[0][:60] if c.snippet.strip() else ""
        print(f"  [{c.symbol_type:17}] {c.chunk_id:40} {c.symbol_name or '':20} {preview}")
    if len(all_chunks) > 15:
        print(f"  ... and {len(all_chunks) - 15} more")