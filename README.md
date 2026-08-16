# RepoQuery

RepoQuery is a high-performance Retrieval-Augmented Generation (RAG) system and codebase search engine. It transforms source code repositories into searchable vector and keyword indices, enabling  natural language code exploration and real-time streaming Q&A powered by Google Gemini

---

## Architecture & Step-by-Step Flow

### Step 1: Repository Ingestion & Scanning
- **Component**: Repository Scanner (`ingestion/scanner.py`)
- **How it works**:
  - Dynamically parses `.gitignore` rules using `pathspec` to exclude build outputs, virtual environments, and external packages (`node_modules`, `dist`, `.venv`).
  - Inspects file headers for null-byte binary signatures to filter out images and compiled assets.
  - Reads text using fallback decoding (`UTF-8` -> `Latin-1` -> lossy replacement) so no valid source file is skipped.

### Step 2: Structure-Aware Code Chunking
- **Component**: Multi-Language Chunker (`ingestion/chunker.py` and `ingestion/main.py`)
- **How it works**:
  - Uses Python AST parsing and boundary regex matchers for TypeScript, JavaScript, Go, Rust, Java, C/C++, and SQL.
  - Enforces a 120-line maximum line ceiling per chunk using a 25-line sliding window overlap to guarantee clean context boundaries.
  - Aggregates chunk records and writes structured output to `data/chunks.json`.

### Step 3: Vector Embedding & Storage
- **Component**: Vector Embedder (`embeddings/embedder.py`)
- **How it works**:
  - Transforms chunk snippets and metadata headers into 384-dimensional normalized vector embeddings using `all-MiniLM-L6-v2`.
  - Persists vector arrays to `embeddings.npy` and manifest metadata to `embedding_manifest.json`.

### Step 4: Hybrid Search & BM25 Caching
- **Component**: Hybrid Retriever (`retrieval/index.py`)
- **How it works**:
  - Tokenizes code identifiers, splitting `camelCase` and `snake_case` symbols for precise keyword matching.
  - Checks for cached `bm25.pkl` for instant index loading; builds and serializes it if missing.
  - Merges dense FAISS scores and sparse BM25 scores via Reciprocal Rank Fusion (RRF).

### Step 5: Context Assembly & Streaming Generation
- **Component**: RAG Answer Engine (`llm/prompt.py`, `llm/gemini_client.py`, `llm/answer.py`)
- **How it works**:
  - Constructs a strict context prompt mandating factual line citations in `file_path:start_line-end_line` format.
  - Uses `gemini-3.6-flash` (with fallback to `gemini-3.5-flash` and `gemini-flash-latest`) with instant 404 error skipping.
  - Stream answers in real-time token chunks via `generate_answer_stream` for instant feedback.

### Step 6: Web Dashboard & CLI Interface
- **Component**: Web Dashboard (`app.py`) & Unified CLI (`repoquery.py`)
- **How it works**:
  - **Web Dashboard (`app.py`)**: Built with Streamlit, providing real-time chat streaming, expandable citations, symbol keyword search, and repository index analytics. Uses `@st.cache_resource` to keep models and FAISS vectors cached in memory.
  - **Unified CLI (`repoquery.py`)**: Supports single-question evaluation (`ask`), interactive terminal REPL, pure code search (`search`), and repository pre-indexing (`index`).

---

## Quick Start

### 1. Installation
```bash
MAC

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


WINDOWS

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. How to Use

#### Option A: Web Dashboard UI (Streamlit)
Launch the visual web interface in your browser:
```bash
streamlit run app.py (this step might take some time)
```
- Open `http://localhost:8501`
- Enter your repository directory path in the sidebar
- Click "Build / Re-index Repository" if not indexed yet
- Use the **Code QA** tab to ask questions with live token streaming and expandable citations
- Use the **Code Search** tab for instant symbol and keyword search (works without GeminiAPI)
- Use the **Index Analytics** tab to view language breakdowns and indexed chunk tables
- Ensure the file path is without inverted commas -> /Users/roystonsoans/Neon Survivor
