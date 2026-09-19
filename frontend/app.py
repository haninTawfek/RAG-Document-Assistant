"""Streamlit chat interface for the RAG Document Assistant."""
import time
import streamlit as st
from api_client import API_BASE_URL, APIError, ask, check_health, upload_file

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Document Assistant", 
    page_icon="📄", 
    layout="centered"
)

# --- Helper Functions ---
def get_relevance_score(score: float) -> tuple[int, str]:
    """
    تحويل قيمة الـ distance إلى نسبة مئوية وتحديد مؤشر جودة المطابقة.
    """
    try:
        score_val = float(score)
        relevance = max(0, min(100, int((1 - score_val) * 100)))
    except (ValueError, TypeError):
        return 0, "⚪ N/A"

    if relevance >= 70:
        indicator = "🟢 High Match"
    elif relevance >= 40:
        indicator = "🟡 Medium Match"
    else:
        indicator = "🔴 Low Match"

    return relevance, indicator


def render_sources(chunks: list):
    """
    دالة مخصصة لتنسيق وعرض قائمة المصادر بشكل دقيق وجذاب.
    """
    with st.expander(f"📚 Sources ({len(chunks)})"):
        for i, chunk in enumerate(chunks, start=1):
            score = chunk.get("score", 0)
            relevance_pct, indicator = get_relevance_score(score)
            label = chunk.get("label", "Document")
            preview = chunk.get("preview", "")

            st.markdown(
                f"**[{i}] {label}** &nbsp;|&nbsp; "
                f"**{indicator} ({relevance_pct}%)**"
            )
            st.info(f'"{preview}..."')


# --- Sidebar: Connection Status & File Upload ---
with st.sidebar:
    st.header("⚙️ Status")
    st.caption(f"Backend: `{API_BASE_URL}`")
    try:
        health = check_health()
        if health.get("vector_store_loaded"):
            st.success(f"Connected · {health.get('indexed_chunks', 0)} chunks indexed")
        else:
            st.warning("Backend is up but the vector store failed to load.")
            
        if not health.get("llm_available"):
            st.warning(f"Ollama/LLM model `{health.get('model')}` not found.")
    except APIError as exc:
        st.error(f"Connection Error: {exc}")

    st.divider()

    # --- Upload Document Section ---
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Upload to Vector Store", use_container_width=True, type="primary"):
            progress_bar = st.progress(0, text="Starting document processing...")
            status_text = st.empty()
            
            try:
                # محاكاة مرحلة القراءة والرفع
                progress_bar.progress(25, text="📖 Reading document content...")
                file_bytes = uploaded_file.getvalue()
                
                progress_bar.progress(50, text="⚡ Sending to Backend & generating embeddings...")
                res = upload_file(file_bytes, uploaded_file.name)
                
                progress_bar.progress(100, text="✅ Indexing complete!")
                time.sleep(0.5)
                
                # إزالة شريط التقدم وإظهار إشعار النجاح
                progress_bar.empty()
                status_text.empty()
                st.toast("✅ File uploaded and indexed successfully!", icon="🎉")
                
                # إعادة تحميل الصفحة لتحديث عدد الـ chunks في Status
                st.rerun()
                
            except APIError as exc:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Upload failed: {exc}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Unexpected error: {e}")

    st.divider()

    # --- Clear Conversation Section ---
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Interface ---
st.title("📄 RAG Document Assistant")
st.caption("Answers are generated strictly from the indexed documents with exact citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chunks"):
            render_sources(message["chunks"])

# إدخال السؤال من المستخدم
question = st.chat_input("Ask a question about the documents...")

if question:
    # 1. عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. تجهيز الـ History
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

    # 3. إرسال الطلب واستقبال الإجابة
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = ask(question, history=history)
            except APIError as exc:
                st.error(f"API Error: {exc}")
                result = None

        if result:
            st.markdown(result["answer"])
            
            if result.get("chunks"):
                render_sources(result["chunks"])
                
            st.caption(f"⚡ {result.get('model', 'LLM')} · {result.get('elapsed_ms', 0)} ms")
            
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "chunks": result.get("chunks", []),
                }
            )