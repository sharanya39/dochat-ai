import streamlit as st
import os
import tempfile
import json
import base64
from rag_engine import ingest_document, ask_question, clear_session_docs
from langchain_core.messages import HumanMessage, AIMessage
from chat_history import (
    save_message, load_messages,
    save_uploaded_doc, load_uploaded_docs,
    clear_session_history
)
from streamlit_oauth import OAuth2Component
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="📄",
    layout="wide",
)

# ── Google OAuth Setup ────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

oauth2 = OAuth2Component(
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    "https://accounts.google.com/o/oauth2/auth",
    "https://oauth2.googleapis.com/token",
    "https://oauth2.googleapis.com/token",
    None,
)

# ── Login Page ────────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.title("📄 DocChat AI")
    st.markdown("### Chat with your documents using Agentic AI")
    st.divider()
    st.markdown("#### Sign in to get started")
    redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8501")
    result = oauth2.authorize_button(
        name="Continue with Google",
        icon="https://www.google.com.tw/favicon.ico",
        redirect_uri=redirect_uri,
        scope="openid email profile",
        key="google",
        extras_params={"prompt": "consent", "access_type": "offline"},
        use_container_width=False,
    )
    if result and "token" in result:
        st.session_state.token = result["token"]
        st.rerun()
    st.stop()

# ── Decode user info from token ───────────────────────────────────────────────
token = st.session_state.token
id_token = token.get("id_token", "")

try:
    # Decode JWT payload (middle part)
    payload = id_token.split(".")[1]
    # Add padding if needed
    payload += "=" * (4 - len(payload) % 4)
    user_info = json.loads(base64.urlsafe_b64decode(payload))
except Exception:
    st.error("Failed to decode user info. Please log in again.")
    del st.session_state.token
    st.rerun()

user_id = user_info.get("sub")
user_email = user_info.get("email")
user_name = user_info.get("name", "User")
user_picture = user_info.get("picture", "")

st.title("📄 DocChat AI")
st.caption("Upload documents and chat with them using Agentic RAG")

# ── Load persisted state from MongoDB on first load ───────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    saved_messages = load_messages(user_id)
    st.session_state.chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "human"
        else AIMessage(content=m["content"])
        for m in saved_messages
    ]
    st.session_state.documents_uploaded = load_uploaded_docs(user_id)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # User profile
    col1, col2 = st.columns([1, 3])
    with col1:
        if user_picture:
            st.image(user_picture, width=45)
    with col2:
        st.markdown(f"**{user_name}**")
        st.caption(user_email)

    if st.button("Logout", type="secondary"):
        del st.session_state.token
        cookie_manager.delete("dochat_token")
        st.session_state.initialized = False
        st.session_state.chat_history = []
        st.session_state.documents_uploaded = []
        st.rerun()

    st.divider()
    st.header("📁 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Your documents are private and tied to your account",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents_uploaded:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        chunk_count, already_existed = ingest_document(tmp_path, user_id, uploaded_file.name)
                        st.session_state.documents_uploaded.append(uploaded_file.name)
                        save_uploaded_doc(user_id, uploaded_file.name)
                        if already_existed:
                            st.info(f"📂 {uploaded_file.name} already exists. Fetching from shared knowledge base.")
                        else:
                            st.success(f"✅ {uploaded_file.name} ingested ({chunk_count} chunks).")
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
            clear_session_docs(user_id)
            clear_session_history(user_id)
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
                session_id=user_id,
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
    save_message(user_id, "human", question)
    save_message(user_id, "ai", result["answer"])
