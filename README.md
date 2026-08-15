# RepoQuery

RepoQuery is a high-performance, local-first Retrieval-Augmented Generation (RAG) system and codebase search engine. It transforms source code repositories into searchable vector and keyword indices, enabling precise natural language code exploration and cited Q&A powered by Google Gemini.

---

## Core Concepts

- **BM25 (Sparse Keyword Search)**: Ranks code snippets by matching exact token frequencies like variable names, function identifiers, or API endpoints.
- **FAISS & Dense Embeddings**: Maps code semantics into 384-dimensional vector space (`all-MiniLM-L6-v2`).
- **Reciprocal Rank Fusion (RRF)**: A rank-blending algorithm that mathematically combines dense semantic search and sparse keyword search into a precision score.
- **AST & Structural Chunking**: Parses code structures like functions, classes, and methods rather than splitting arbitrarily, preserving logic boundaries.
- **Sliding Window Safeguard**: Automatically splits oversized files into overlapping 120-line windows to prevent context window overflow during LLM inference.

---

## Architecture & Step-by-Step Flow

### Step 1: Repository Ingestion & Scanning
- **Component**: Repository Scanner (`ingestion/scanner.py`)
- **How it works**:
  - Dynamically parses `.gitignore` rules to exclude build outputs, virtual environments, and dependencies (`node_modules`, `dist`, `.venv`).
  - Inspects file headers for null-byte binary signatures to automatically filter out images and compiled assets.
  - Reads text using fallback decoding (`UTF-8` $\rightarrow$ `Latin-1` $\rightarrow$ lossy replacement) so no valid source file is dropped.

### Step 2: Structure-Aware Code Chunking
- **Component**: Multi-Language Chunker (`ingestion/chunker.py` and `ingestion/main.py`)
- **How it works**:
  - Uses AST parsing for Python and regex boundary matchers for TypeScript, JavaScript, Go, Rust, Java, C/C++, and SQL.
  - Enforces a 120-line maximum line ceiling per chunk using a 25-line sliding window overlap to guarantee clean context boundaries.
  - Aggregates chunk records and writes structured output to `data/chunks.json`.

### Step 3: Vector Embedding & Storage
- **Component**: Vector Embedder (`embeddings/embedder.py`)
- **How it works**:
  - Transforms chunk snippets and metadata headers into 384-dimensional normalized vector embeddings.
  - Persists vector arrays to `embeddings.npy` and manifest data to `embedding_manifest.json`.

### Step 4: Hybrid Search & Rank Fusion
- **Component**: Hybrid Retriever (`retrieval/index.py`)
- **How it works**:
  - Tokenizes code by splitting `camelCase`, `snake_case`, and identifier symbols for precise term matching.
  - Executes parallel searches across the FAISS vector index (dense) and BM25 index (sparse).
  - Merges result lists using RRF scoring: $\text{RRF}(d) = \frac{0.6}{60 + \text{rank}_{\text{dense}}} + \frac{0.4}{60 + \text{rank}_{\text{bm25}}}$.

### Step 5: Context Assembly & LLM Generation
- **Component**: RAG Answer Engine (`llm/prompt.py`, `llm/gemini_client.py`, `llm/answer.py`)
- **How it works**:
  - Constructs a strict context prompt mandating factual citations in `file_path:start_line-end_line` format.
  - Invokes Google Gemini models (`gemini-flash-latest` cascade) with exponential backoff retries and disabled automatic function calling warning logs.

### Step 6: CLI Orchestration & Caching
- **Component**: Command-Line Interface (`repoquery.py`)
- **How it works**:
  - Computes deterministic path hashes for indexed repositories (`data/repo_name_hash/`).
  - Reuses existing index caches for instant responses on subsequent queries.
  - Provides interactive REPL sessions and standalone code search modes.

---

## Quick Start

### 1. Installation
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Running Options

#### Option A: Ask a Question (RAG with Gemini Answer)
```bash
python repoquery.py ask /path/to/target/repository "Where are questions generated?"
```

#### Option B: Interactive Chat Mode (Ask Multiple Questions)
```bash
python repoquery.py ask /path/to/target/repository
```

#### Option C: Fast Code Search (No LLM Tokens Used)
```bash
python repoquery.py search /path/to/target/repository "GEMINI_MODELS"
```

#### Option D: Explicitly Pre-Index a Repository
```bash
python repoquery.py index /path/to/target/repository
```

