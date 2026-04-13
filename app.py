import streamlit as st
import os
import uuid
import tempfile
from rag_engine import ingest_document, ask_question, clear_session_docs
from langchain_core.messages import HumanMessage, AIMessage
from chat_history import (
    save_message, load_messages,
    save_uploaded_doc, load_uploaded_docs,
    clear_session_history
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocChat AI")
st.caption("Upload documents and chat with them using Agentic RAG")

# ── Stable Session ID (persists across refreshes via URL params) ──────────────
if "session_id" not in st.query_params:
    st.query_params["session_id"] = str(uuid.uuid4())

session_id = st.query_params["session_id"]

# ── Load persisted state from MongoDB on first load ───────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    # Restore chat history
    saved_messages = load_messages(session_id)
    st.session_state.chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "human"
        else AIMessage(content=m["content"])
        for m in saved_messages
    ]

    # Restore uploaded documents list
    st.session_state.documents_uploaded = load_uploaded_docs(session_id)

# ── Sidebar: Document Upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Your Documents")
    st.caption(f"Session: `{session_id[:8]}...`")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each user session is isolated — your documents are private",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents_uploaded:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        chunk_count, already_existed = ingest_document(tmp_path, session_id, uploaded_file.name)
                        st.session_state.documents_uploaded.append(uploaded_file.name)
                        save_uploaded_doc(session_id, uploaded_file.name)
                        if already_existed:
                            st.info(f"📂 {uploaded_file.name} already exists in the database. Fetching from shared knowledge base — no re-ingestion needed.")
                        else:
                            st.success(f"✅ {uploaded_file.name} ingested successfully ({chunk_count} chunks stored in MongoDB).")
                    except Exception as e:
                        import traceback
                        st.error(f"Ingestion failed: {e}")
                        st.code(traceback.format_exc())
                    finally:
                        os.unlink(tmp_path)

    if st.session_state.documents_uploaded:
        st.divider()
        st.subheader("Loaded Documents")
        for doc in st.session_state.documents_uploaded:
            st.write(f"• {doc}")

        if st.button("🗑️ Clear All Documents", type="secondary"):
            clear_session_docs(session_id)
            clear_session_history(session_id)
            st.session_state.documents_uploaded = []
            st.session_state.chat_history = []
            st.session_state.initialized = False
            st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload PDF(s)
    2. Ask questions in chat
    3. AI retrieves relevant chunks
    4. **Agentic re-retrieval** kicks in if needed
    5. Claude synthesizes the answer
    """)

# ── Chat Interface ────────────────────────────────────────────────────────────
if not st.session_state.documents_uploaded:
    st.info("👈 Upload a PDF document to get started")
    st.stop()

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if question := st.chat_input("Ask anything about your documents..."):
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(
                question=question,
                session_id=session_id,
                chat_history=st.session_state.chat_history,
            )

        st.write(result["answer"])

        if result["used_reretrieval"]:
            st.caption("🔄 Agentic re-retrieval was used to improve this answer")

        if result["sources"]:
            with st.expander("📚 Sources used"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}** — {src['source']} (Page {src['page']})")
                    st.text(src["preview"])

    # Save to session state and MongoDB
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))
    save_message(session_id, "human", question)
    save_message(session_id, "ai", result["answer"])
