import streamlit as st
import os
import uuid
import tempfile
from rag_engine import ingest_document, ask_question, clear_session_docs
from langchain_core.messages import HumanMessage, AIMessage

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocChat AI")
st.caption("Upload documents and chat with them using Agentic RAG")

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = []

session_id = st.session_state.session_id

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
                        from pymongo import MongoClient
                        from rag_engine import collection
                        already_exists = collection.count_documents({"source": uploaded_file.name}) > 0
                        chunk_count = ingest_document(tmp_path, session_id, uploaded_file.name)
                        st.session_state.documents_uploaded.append(uploaded_file.name)
                        if already_exists:
                            st.info(f"📂 {uploaded_file.name} was already uploaded before. Retrieving results from stored MongoDB data ({chunk_count} chunks).")
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
            st.session_state.documents_uploaded = []
            st.session_state.chat_history = []
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
    # Show user message
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(
                question=question,
                session_id=session_id,
                chat_history=st.session_state.chat_history,
            )

        st.write(result["answer"])

        # Show agentic badge if re-retrieval was used
        if result["used_reretrieval"]:
            st.caption("🔄 Agentic re-retrieval was used to improve this answer")

        # Expandable sources
        if result["sources"]:
            with st.expander("📚 Sources used"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}** — {src['source']} (Page {src['page']})")
                    st.text(src["preview"])

    # Update history
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))
