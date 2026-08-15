import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from repoquery import get_default_data_dir, is_index_ready, index_repo
from retrieval.index import Retriever
from llm.answer import ask_question

# Page Configuration
st.set_page_config(
    page_title="RepoQuery - Codebase QA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #F3F4F6;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        padding: 0 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_repo" not in st.session_state:
        st.session_state.current_repo = ""


@st.cache_resource(show_spinner=False)
def get_retriever(data_dir: str) -> Retriever:
    """Cached retriever instance stored in memory across user interactions."""
    return Retriever(data_dir)



def main():
    init_session()

    # Sidebar Controls
    with st.sidebar:
        st.title("RepoQuery")
        st.caption("Codebase RAG & Hybrid Search Engine")
        
        repo_path = st.text_input(
            "Repository Path",
            value="/Users/roystonsoans/Pokelearn",
            help="Enter absolute path to the target repository"
        ).strip()

        data_dir = get_default_data_dir(repo_path) if repo_path else ""
        indexed = is_index_ready(data_dir) if data_dir else False

        # Status & Metrics
        if indexed:
            st.success("Index Ready")
            chunks_file = os.path.join(data_dir, "chunks.json")
            if os.path.isfile(chunks_file):
                with open(chunks_file, "r") as f:
                    chunk_data = json.load(f)
                num_chunks = len(chunk_data)
                num_files = len(set(c["file_path"] for c in chunk_data))
                
                col1, col2 = st.columns(2)
                col1.metric("Files", num_files)
                col2.metric("Chunks", num_chunks)
        else:
            st.warning("Index Not Built")

        st.markdown("---")
        st.subheader("Indexing")

        if st.button("Build / Re-index Repository", use_container_width=True):
            if not repo_path or not os.path.isdir(repo_path):
                st.error("Invalid repository directory path!")
            else:
                with st.spinner("Scanning and indexing repository..."):
                    try:
                        index_repo(repo_path, data_dir, force=True)
                        st.success("Indexing complete")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

        st.markdown("---")
        st.subheader("Settings")
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=15, value=5)
        selected_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"],
            index=0
        )

        st.markdown("---")
        st.caption("FAISS Vector + BM25 Sparse Search | Google Gemini AI")

    # Main Dashboard
    st.markdown('<div class="main-header">RepoQuery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Codebase Question Answering and Hybrid Vector Search</div>', unsafe_allow_html=True)

    if not repo_path or not os.path.isdir(repo_path):
        st.info("Please enter a valid repository directory path in the sidebar to begin.")
        return

    if not indexed:
        st.warning(f"Repository {repo_path} has not been indexed yet.")
        if st.button("Build Index Now"):
            with st.spinner("Building index..."):
                index_repo(repo_path, data_dir, force=True)
                st.rerun()
        return

    # Tabs Interface (No emojis)
    tab_chat, tab_search, tab_analytics = st.tabs(["Code QA", "Code Search", "Index Analytics"])

    # Load Retriever
    try:
        retriever = get_retriever(data_dir)
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return

    # TAB 1: Chat QA
    with tab_chat:
        st.caption("Ask questions about the codebase. Answers include precise file and line citations.")
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("Sources & Citations"):
                        for c in msg["citations"]:
                            sym = f" (`{c['symbol_name']}`)" if c.get("symbol_name") else ""
                            st.markdown(f"**Score {c['score']:.4f}** — `{c['file_path']}:{c['start_line']}-{c['end_line']}`{sym}")

        # Chat Input
        if question := st.chat_input("Ask a question about the code..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating response..."):
                    try:
                        result = ask_question(retriever, question, top_k=top_k)
                        answer = result["answer"]
                        citations = result["citations"]
                        
                        st.markdown(answer)
                        if citations:
                            with st.expander("Sources & Citations", expanded=True):
                                for c in citations:
                                    sym = f" (`{c['symbol_name']}`)" if c.get("symbol_name") else ""
                                    st.markdown(f"**Score {c['score']:.4f}** — `{c['file_path']}:{c['start_line']}-{c['end_line']}`{sym}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

    # TAB 2: Hybrid Search
    with tab_search:
        st.subheader("Code & Symbol Search")
        st.caption("Search variable names, function identifiers, or concepts via FAISS + BM25 hybrid search.")
        
        search_query = st.text_input("Search query or symbol name...", value="", key="search_input")
        if search_query:
            results = retriever.search(search_query, top_k=top_k)
            st.markdown(f"**Matches for `{search_query}` ({len(results)})**")
            
            for idx, r in enumerate(results):
                sym = f" — `{r['symbol_type']} {r.get('symbol_name') or ''}`" if r.get('symbol_name') else ""
                with st.container():
                    st.markdown(f"**{idx + 1}. `{r['file_path']}:{r['start_line']}-{r['end_line']}`**{sym}")
                    st.caption(f"Fusion Score: {r['score']:.4f} | Dense Score: {r.get('dense_score', 0):.4f} | BM25 Score: {r.get('bm25_score', 0):.4f}")
                    
                    lang = r.get("language", "python")
                    st.code(r["snippet"], language=lang)
                    st.markdown("---")

    # TAB 3: Analytics
    with tab_analytics:
        st.subheader("Index Analytics")
        chunks_file = os.path.join(data_dir, "chunks.json")
        if os.path.isfile(chunks_file):
            with open(chunks_file, "r") as f:
                chunks = json.load(f)
            
            df = pd.DataFrame(chunks)
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("##### Languages")
                lang_counts = df["language"].value_counts().reset_index()
                lang_counts.columns = ["Language", "Chunks"]
                st.bar_chart(lang_counts.set_index("Language"))

            with col_b:
                st.markdown("##### Symbol Types")
                type_counts = df["symbol_type"].value_counts().reset_index()
                type_counts.columns = ["Symbol Type", "Count"]
                st.dataframe(type_counts, use_container_width=True)

            st.markdown("##### Indexed Chunks Manifest")
            st.dataframe(
                df[["file_path", "start_line", "end_line", "language", "symbol_type", "symbol_name"]],
                use_container_width=True
            )


if __name__ == "__main__":
    main()

