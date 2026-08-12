import os
from dataclasses import dataclass, field
from typing import List

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".md": "markdown",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".mypy_cache", ".pytest_cache",
    "site-packages", ".idea", ".vscode",
}

MAX_FILE_SIZE_BYTES = 500_000

@dataclass
class FileRecord:
    file_path: str       
    absolute_path: str
    language: str
    size_bytes: int
    content: str


def scan_repository(repo_root: str) -> List[FileRecord]:
    if not os.path.isdir(repo_root):
        raise ValueError(f"Not a directory: {repo_root}")

    records: List[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune ignored directories in place
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower() #just get the extension
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            absolute_path = os.path.join(dirpath, filename)

            try:
                size = os.path.getsize(absolute_path)
            except OSError:
                continue

            if size == 0 or size > MAX_FILE_SIZE_BYTES:
                continue

            content = _read_text_file(absolute_path)
            if content is None:
                continue  

            relative_path = os.path.relpath(absolute_path, repo_root)

            records.append(FileRecord(
                file_path=relative_path,
                absolute_path=absolute_path,
                language=SUPPORTED_EXTENSIONS[ext],
                size_bytes=size,
                content=content,
            ))

    return records


def _read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


if __name__ == "__main__":
    import sys
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    results = scan_repository(target)

    print(f"Scanned {target}: found {len(results)} supported files\n")
    for r in results[:20]:
        print(f"  [{r.language:10}] {r.file_path}  ({r.size_bytes} bytes)")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")