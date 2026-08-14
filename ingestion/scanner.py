import os
from dataclasses import dataclass
from typing import List, Optional

try:
    import pathspec
except ImportError:
    pathspec = None

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sql": "sql",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "out", ".next", ".nuxt", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", ".idea", ".vscode", "target", "bin", "obj",
    ".turbo", ".cache", "coverage", ".docusaurus"
}

MAX_FILE_SIZE_BYTES = 1_000_000  # 1MB limit


@dataclass
class FileRecord:
    file_path: str       
    absolute_path: str
    language: str
    size_bytes: int
    content: str


def _load_gitignore(repo_root: str):
    """Load and compile .gitignore from repo_root if present."""
    if not pathspec:
        return None

    gitignore_path = os.path.join(repo_root, ".gitignore")
    patterns = []
    if os.path.isfile(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                patterns.extend(f.read().splitlines())
        except OSError:
            pass

    # Add standard global ignores to pathspec
    patterns.extend([f"**/{d}/" for d in IGNORED_DIRS])
    patterns.extend([f"{d}/" for d in IGNORED_DIRS])
    
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def scan_repository(repo_root: str) -> List[FileRecord]:
    if not os.path.isdir(repo_root):
        raise ValueError(f"Not a directory: {repo_root}")

    repo_root = os.path.abspath(repo_root)
    spec = _load_gitignore(repo_root)
    records: List[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Calculate relative dir path from repo_root
        rel_dir = os.path.relpath(dirpath, repo_root)
        if rel_dir == ".":
            rel_dir = ""

        # Filter ignored directories
        pruned_dirs = []
        for d in dirnames:
            if d in IGNORED_DIRS or d.startswith("."):
                continue
            dir_rel_path = os.path.join(rel_dir, d) if rel_dir else d
            # Check gitignore for directory (with trailing slash)
            if spec and spec.match_file(f"{dir_rel_path}/"):
                continue
            pruned_dirs.append(d)
        dirnames[:] = pruned_dirs

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            absolute_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(absolute_path, repo_root)

            # Check gitignore match
            if spec and spec.match_file(rel_path):
                continue

            try:
                size = os.path.getsize(absolute_path)
            except OSError:
                continue

            if size == 0 or size > MAX_FILE_SIZE_BYTES:
                continue

            content = _read_text_file(absolute_path)
            if content is None:
                continue

            records.append(FileRecord(
                file_path=rel_path,
                absolute_path=absolute_path,
                language=SUPPORTED_EXTENSIONS[ext],
                size_bytes=size,
                content=content,
            ))

    return records


def _read_text_file(path: str) -> Optional[str]:
    """Read a text file with robust encoding fallback and null-byte detection."""
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Check for binary file
        if b"\x00" in raw_bytes[:1024]:
            return None

        # Try UTF-8 first
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pass

        # Try Latin-1 or fallback with replacement chars
        try:
            return raw_bytes.decode("latin-1")
        except UnicodeDecodeError:
            return raw_bytes.decode("utf-8", errors="replace")
    except OSError:
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