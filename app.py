"""
Streamlit Web App for Modular RAG Pipeline - Modern Gemini/ChatGPT UI Edition
"""
import os
import streamlit as st

# Modular package imports
from modules import (
    load_text_from_file, get_document_stats,
    recursive_chunks, semantic_chunks, fixed_size_chunks,
    get_openai_client, embed, cosine_similarity,
    build_faiss_index, run_rag_query,
    load_custom_css, init_session_state
)

# App Setup & Modern CSS
st.set_page_config(page_title="PDF Detective", page_icon="🔍", layout="wide")
load_custom_css("static/style.css")
init_session_state()

# ----------------- SIDEBAR SETTINGS -----------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    provider = st.selectbox("API Provider", ["OpenRouter", "OpenAI"])
    default_url = "https://openrouter.ai/api/v1" if provider == "OpenRouter" else "https://api.openai.com/v1"
    
    api_key = st.text_input(
        f"{provider} API Key", 
        value=os.getenv("OPENROUTER_API_KEY" if provider == "OpenRouter" else "OPENAI_API_KEY", ""),
        type="password",
        help="Reads automatically from .env or you can paste directly here"
    )
    base_url = st.text_input("Base URL", value=default_url)

    st.markdown("---")
    st.markdown("### 🧠 Model & Chunking")
    embed_model = "openai/text-embedding-3-small" if provider == "OpenRouter" else "text-embedding-3-small"
    chat_models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-haiku",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-7b-instruct:free"
    ] if provider == "OpenRouter" else ["gpt-4o-mini", "gpt-4o"]
    chat_model = st.selectbox("Chat Model", chat_models)

    chunk_method = st.selectbox("Chunking Strategy", ["Recursive (LangChain)", "Semantic (Paragraph)", "Fixed Size"])
    chunk_size = st.slider("Chunk Size (Characters)", 200, 1500, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top K Retrieved Chunks", 1, 6, 3)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ----------------- HERO HEADER -----------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-left">
        <div class="hero-title-row">
            <div class="hero-spark-badge">🔍</div>
            <h1 class="main-header">PDF Detective</h1>
        </div>
        <p class="sub-header">Document Intelligence • FAISS Neural Vector Indexing • Grounded Citations</p>
    </div>
    <div class="model-pill-badge">
        <span>⚡</span>
        <span>{chat_model}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- CLEAN TABS -----------------
tab_doc, tab_chat, tab_sim = st.tabs([
    "📄 Document & Indexing",
    "💬 RAG Chat Assistant",
    "🔬 Semantic Similarity"
])

# ----------------- TAB 1: DOCUMENT & INDEXING -----------------
with tab_doc:
    col1, col2 = st.columns([1.6, 1.1], gap="medium")
    with col1:
        st.markdown("##### 📥 Document Ingestion")
        source = st.radio("Input Source:", ["Upload File (.pdf, .txt)", "Paste Text"], horizontal=True)
        if source == "Upload File (.pdf, .txt)":
            file = st.file_uploader("Upload your document", type=["txt", "pdf"], help="Supports PDF and plain text (.txt)")
            if file:
                st.session_state["raw_text"] = load_text_from_file(file)
        else:
            pasted = st.text_area(
                "Paste your text here:",
                height=210,
                placeholder="Paste articles, study notes, policies, or technical documentation here..."
            )
            if pasted:
                st.session_state["raw_text"] = pasted

    with col2:
        st.markdown("##### 📊 Document Overview")
        if st.session_state["raw_text"]:
            stats = get_document_stats(st.session_state["raw_text"])
            num_chunks = len(st.session_state["chunks"])
            vector_count = st.session_state["index"].ntotal if st.session_state["index"] else 0
            
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-item">
                    <span class="stat-item-label">Characters</span>
                    <span class="stat-item-value">{stats['chars']:,}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-item-label">Words</span>
                    <span class="stat-item-value">{stats['words']:,}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-item-label">Chunks</span>
                    <span class="stat-item-value">{num_chunks:,}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-item-label">Vectors</span>
                    <span class="stat-item-value">{vector_count:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state["index"]:
                st.markdown(f"""
                <div class="status-pill ready">
                    <span class="pulse-dot"></span>
                    <span>FAISS Index Active • {vector_count} vectors indexed</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-pill waiting">
                    <span>⏳ Ready to Index • Click the button below</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-panel" style="text-align: center; color: #64748B; padding: 2.3rem 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <div style="font-weight: 600; color: #334155; font-size: 1rem;">No document loaded</div>
                <div style="font-size: 0.85rem; margin-top: 0.35rem; color: #64748B;">Upload a PDF/TXT or paste text on the left to index into FAISS.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("✨ Process & Build FAISS Vector Index", type="primary", use_container_width=True):
        if not st.session_state["raw_text"]:
            st.error("Pehle document upload ya paste karein!")
        elif not api_key:
            st.error(f"Sidebar me {provider} API key enter karein.")
        else:
            with st.spinner("⚡ Chunking, Embeddings generation, aur FAISS Indexing in progress..."):
                txt = st.session_state["raw_text"]
                if chunk_method == "Recursive (LangChain)":
                    chunks = recursive_chunks(txt, chunk_size, chunk_overlap)
                elif chunk_method == "Semantic (Paragraph)":
                    chunks = semantic_chunks(txt, chunk_size)
                else:
                    chunks = fixed_size_chunks(txt, chunk_size, chunk_overlap)

                client = get_openai_client(api_key, base_url)
                vectors = embed(chunks, client, embed_model)
                st.session_state["chunks"] = chunks
                st.session_state["index"] = build_faiss_index(vectors)
                st.toast("🎉 FAISS Indexing complete!", icon="✅")
                st.success(f"🎉 Success! {len(chunks)} chunks bane aur FAISS index taiyar ho gaya.")

# ----------------- TAB 2: RAG CHAT ASSISTANT -----------------
with tab_chat:
    if not st.session_state["index"]:
        st.markdown("""
        <div class="card-panel" style="text-align: center; padding: 2rem; color: #64748B;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔒</div>
            <h4 style="color: #1E293B; margin-bottom: 0.25rem;">FAISS Index Not Found</h4>
            <p style="margin: 0; font-size: 0.9rem;">Pehle <b>'Document & Indexing'</b> tab me jakar document process karein taaki vector search enable ho sake.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not st.session_state["history"]:
            st.markdown("""
            <div class="card-panel" style="text-align: center; padding: 2.2rem 1.5rem; background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">✨</div>
                <h3 style="margin: 0; font-weight: 700; color: #0F172A; font-size: 1.25rem;">Ask anything about your document</h3>
                <p style="color: #64748B; font-size: 0.9rem; margin-top: 0.35rem; margin-bottom: 1.25rem;">
                    Your questions are matched against FAISS vectors and answered with grounded, hallucination-free citations.
                </p>
                <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
                    <span class="prompt-chip">📌 Summarize key concepts</span>
                    <span class="prompt-chip">🎯 What are the core components?</span>
                    <span class="prompt-chip">💡 Explain like I'm five</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        q = st.chat_input("Ask a question about your document...")
        if q:
            if not api_key:
                st.error("API Key missing hai. Kripya sidebar me key check karein.")
            else:
                with st.spinner("Searching FAISS vectors & synthesizing verified answer..."):
                    client = get_openai_client(api_key, base_url)
                    res = run_rag_query(q, st.session_state["chunks"], st.session_state["index"], client, embed_model, chat_model, top_k)
                    st.session_state["history"].append({"q": q, "res": res})

        for item in reversed(st.session_state["history"]):
            with st.chat_message("user"):
                st.markdown(f"**{item['q']}**")
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(item["res"]["answer"])
                sources = item["res"].get("sources", [])
                if sources:
                    with st.expander(f"📚 Verified Sources & Citations ({len(sources)})"):
                        for s in sources:
                            st.markdown(f"""
                            <div class="source-box">
                                <span class="source-tag">Source {s['rank']} • L2 Distance: {s['dist']:.4f}</span>
                                <div>{s['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

# ----------------- TAB 3: COSINE SIMILARITY -----------------
with tab_sim:
    st.markdown("##### 🔬 Semantic Cosine Similarity Calculator")
    st.caption("Compare semantic vector closeness of any two sentences in high-dimensional embedding space.")
    
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        t1 = st.text_area("Text 1:", "Machine Learning is an application of Artificial Intelligence.", height=120)
    with c2:
        t2 = st.text_area("Text 2:", "Deep Learning models use artificial neural networks.", height=120)
    
    if st.button("⚡ Calculate Semantic Similarity", type="primary"):
        if not api_key:
            st.error("API Key enter karein.")
        else:
            with st.spinner("Calculating vectors & dot product..."):
                client = get_openai_client(api_key, base_url)
                v = embed([t1, t2], client, embed_model)
                score = cosine_similarity(v[0], v[1])
                
                st.markdown(f"""
                <div class="card-panel" style="margin-top: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: #475569; font-size: 0.9rem;">Cosine Similarity Metric</span>
                        <span style="font-weight: 800; font-size: 1.4rem; color: #2563EB;">{score:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, float(score))))


