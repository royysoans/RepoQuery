import ast
import re
from dataclasses import dataclass
from typing import List, Optional

try:
    from .scanner import FileRecord
except ImportError:
    from scanner import FileRecord


MAX_CHUNK_LINES = 120
OVERLAP_LINES = 25


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
    elif record.language == "go":
        chunks = _chunk_go(record)
    elif record.language == "rust":
        chunks = _chunk_rust(record)
    elif record.language in ("java", "csharp", "cpp", "c"):
        chunks = _chunk_java_csharp_cpp(record)
    elif record.language == "sql":
        chunks = _chunk_sql(record)
    elif record.language == "markdown":
        chunks = _chunk_markdown(record)
    else:
        chunks = []

    if not chunks:
        chunks = _sliding_window_chunks(record)
    else:
        # Enforce max chunk line limit across all chunks
        bounded_chunks = []
        for c in chunks:
            bounded_chunks.extend(_enforce_chunk_bounds(c, record))
        chunks = bounded_chunks

    return chunks


def _enforce_chunk_bounds(chunk: Chunk, record: FileRecord) -> List[Chunk]:
    lines = chunk.snippet.splitlines()
    if len(lines) <= MAX_CHUNK_LINES:
        return [chunk]

    # Split oversized chunk into sliding windows
    sub_chunks = []
    total_lines = len(lines)
    step = MAX_CHUNK_LINES - OVERLAP_LINES
    for i in range(0, total_lines, step):
        chunk_lines = lines[i : i + MAX_CHUNK_LINES]
        if not chunk_lines:
            break
        start_line = chunk.start_line + i
        end_line = start_line + len(chunk_lines) - 1
        snippet = "\n".join(chunk_lines)
        if not snippet.strip():
            continue

        sym_name = f"{chunk.symbol_name} (part {len(sub_chunks)+1})" if chunk.symbol_name else None
        sub_chunks.append(Chunk(
            chunk_id=f"{record.file_path}:{start_line}-{end_line}",
            file_path=record.file_path,
            language=record.language,
            start_line=start_line,
            end_line=end_line,
            symbol_name=sym_name,
            symbol_type=chunk.symbol_type,
            snippet=snippet,
        ))
        if i + MAX_CHUNK_LINES >= total_lines:
            break

    return sub_chunks if sub_chunks else [chunk]


def _sliding_window_chunks(record: FileRecord) -> List[Chunk]:
    lines = record.content.splitlines()
    if not lines:
        return []

    if len(lines) <= MAX_CHUNK_LINES:
        return [Chunk(
            chunk_id=f"{record.file_path}:1-{len(lines)}",
            file_path=record.file_path,
            language=record.language,
            start_line=1,
            end_line=len(lines),
            symbol_name=None,
            symbol_type="module",
            snippet=record.content,
        )]

    chunks = []
    step = MAX_CHUNK_LINES - OVERLAP_LINES
    for i in range(0, len(lines), step):
        window = lines[i : i + MAX_CHUNK_LINES]
        if not window:
            break
        start_line = i + 1
        end_line = i + len(window)
        snippet = "\n".join(window)
        if snippet.strip():
            chunks.append(Chunk(
                chunk_id=f"{record.file_path}:{start_line}-{end_line}",
                file_path=record.file_path,
                language=record.language,
                start_line=start_line,
                end_line=end_line,
                symbol_name=None,
                symbol_type="module",
                snippet=snippet,
            ))
        if i + MAX_CHUNK_LINES >= len(lines):
            break

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

            # If it's a class, also chunk its methods
            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = child.lineno
                        m_end = _get_end_line(child, lines)
                        m_snippet = "\n".join(lines[m_start - 1:m_end])
                        chunks.append(Chunk(
                            chunk_id=f"{record.file_path}:{m_start}-{m_end}",
                            file_path=record.file_path,
                            language=record.language,
                            start_line=m_start,
                            end_line=m_end,
                            symbol_name=f"{node.name}.{child.name}",
                            symbol_type="method",
                            snippet=m_snippet,
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
    r"interface\s+(?P<interface_name>\w+)|"
    r"type\s+(?P<type_name>\w+)\s*=|"
    r"enum\s+(?P<enum_name>\w+)|"
    r"const\s+(?P<const_name>\w+)\s*=\s*(async\s+)?(\([^)]*\)|\w+)\s*=>|"
    r"(static\s+)?(async\s+)?(?P<method_name>\w+)\s*\([^)]*\)\s*\{)",
)
_JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "throw", "try"}


def _chunk_js_ts(record: FileRecord) -> List[Chunk]:
    return _chunk_by_regex(
        record,
        _JS_BOUNDARY_RE,
        _JS_KEYWORDS,
        type_mapping={"class_name": "class", "interface_name": "interface", "type_name": "type", "enum_name": "enum", "method_name": "method", "func_name": "function", "const_name": "function"}
    )


_GO_BOUNDARY_RE = re.compile(
    r"^\s*(func\s+(\([^)]+\)\s+)?(?P<func_name>\w+)|"
    r"type\s+(?P<type_name>\w+)\s+(struct|interface))"
)


def _chunk_go(record: FileRecord) -> List[Chunk]:
    return _chunk_by_regex(
        record,
        _GO_BOUNDARY_RE,
        set(),
        type_mapping={"type_name": "type", "func_name": "function"}
    )


_RUST_BOUNDARY_RE = re.compile(
    r"^\s*(pub\s+)?(async\s+)?"
    r"(fn\s+(?P<func_name>\w+)|"
    r"struct\s+(?P<struct_name>\w+)|"
    r"enum\s+(?P<enum_name>\w+)|"
    r"trait\s+(?P<trait_name>\w+)|"
    r"impl(\s+<[^>]+>)?\s+(?P<impl_name>[\w:]+))"
)


def _chunk_rust(record: FileRecord) -> List[Chunk]:
    return _chunk_by_regex(
        record,
        _RUST_BOUNDARY_RE,
        set(),
        type_mapping={"func_name": "function", "struct_name": "struct", "enum_name": "enum", "trait_name": "trait", "impl_name": "impl"}
    )


_JAVA_CPP_BOUNDARY_RE = re.compile(
    r"^\s*(public\s+|private\s+|protected\s+|static\s+|final\s+|async\s+)*"
    r"(class\s+(?P<class_name>\w+)|"
    r"interface\s+(?P<interface_name>\w+)|"
    r"enum\s+(?P<enum_name>\w+)|"
    r"(struct\s+(?P<struct_name>\w+))|"
    r"(([\w<>\[\]]+)\s+(?P<func_name>\w+)\s*\([^)]*\)\s*(\{)?))"
)
_JAVA_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "throw", "try", "synchronized", "new"}


def _chunk_java_csharp_cpp(record: FileRecord) -> List[Chunk]:
    return _chunk_by_regex(
        record,
        _JAVA_CPP_BOUNDARY_RE,
        _JAVA_KEYWORDS,
        type_mapping={"class_name": "class", "interface_name": "interface", "enum_name": "enum", "struct_name": "struct", "func_name": "function"}
    )


_SQL_BOUNDARY_RE = re.compile(
    r"^\s*CREATE\s+(OR\s+REPLACE\s+)?(TABLE|FUNCTION|PROCEDURE|VIEW|INDEX|TRIGGER)\s+(?P<symbol_name>[\w\.]+)",
    re.IGNORECASE
)


def _chunk_sql(record: FileRecord) -> List[Chunk]:
    return _chunk_by_regex(
        record,
        _SQL_BOUNDARY_RE,
        set(),
        type_mapping={"symbol_name": "sql_definition"}
    )


def _chunk_by_regex(record: FileRecord, pattern: re.Pattern, keywords: set, type_mapping: dict) -> List[Chunk]:
    lines = record.content.splitlines()
    boundaries = []

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            groups = {k: v for k, v in m.groupdict().items() if v}
            if not groups:
                continue

            name = list(groups.values())[0]
            if name in keywords:
                continue

            first_key = list(groups.keys())[0]
            symbol_type = type_mapping.get(first_key, "symbol")

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
    in_code_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            continue

        if not in_code_block:
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