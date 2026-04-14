# DocChat AI

> **Chat with your documents using Agentic AI** — Upload any PDF and get accurate, cited answers powered by Claude AI, MongoDB Atlas Vector Search, and LangGraph. Multi-user ready with Google OAuth authentication.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-red)](https://langchain-ai.github.io/langgraph/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas_Vector_Search-green)](https://mongodb.com/atlas)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b)](https://streamlit.io)
[![Google OAuth](https://img.shields.io/badge/Auth-Google_OAuth_2.0-4285F4)](https://developers.google.com/identity)

---

## Try It Yourself

This app is not publicly hosted to keep API and database costs in check.

Want to try it? **Fork this repo**, add your own credentials in `.env`, and run it locally or deploy it to your own Streamlit Cloud account — the full setup guide is below.

If you'd like to contribute or suggest improvements, feel free to **open a Pull Request**!

---

## Why DocChat AI?

Tired of scrolling through long PDFs to find answers? DocChat AI lets you **ask questions in plain English** and get precise answers with page citations — instantly.

- No more manual searching through documents
- Answers grounded in your actual document — no hallucinations
- Intelligent re-retrieval if the first search isn't good enough
- Each user's documents are fully private and isolated
- **Multi-user support** — sign in with Google, your data stays yours

---

## Features

| Feature | Description |
|---|---|
| Google OAuth 2.0 | Secure sign-in — no passwords needed |
| Multi-user isolation | Each user's documents and chat history are private |
| Shared knowledge base | Same PDF uploaded by multiple users is stored once — saves storage |
| Duplicate prevention | Uploading the same document twice doesn't re-ingest it |
| Agentic re-retrieval | If initial search is weak, Claude generates better queries and retries |
| Chat history persistence | Conversation saved in MongoDB — survives page refresh |
| Source citations | Every answer includes document name and page number |
| In-memory LLM cache | Repeated questions get consistent, fast answers |

---

## Architecture

```
PDF Upload → Hash Check → [Already in DB? Skip] → Chunk → Embed → MongoDB Atlas
                                                                          ↓
Google Login → user_id → User Question → Retrieve (user-scoped) → Evaluate
                                                                          ↓
                                              [Score < 6] → Expand Queries → Re-retrieve
                                                                          ↓
                                                              Generate Answer + Citations
```

### Agentic RAG Pipeline (LangGraph)

1. **Retrieve** — Fetch top-5 relevant chunks from MongoDB Atlas, filtered by `user_id`
2. **Evaluate** — Claude scores retrieval relevance (1–10)
3. **Expand & Re-retrieve** — If score < 6, generate better search queries and re-retrieve
4. **Generate** — Claude synthesizes the final answer with source citations

### Multi-user & Shared Knowledge Base

- Each user authenticates via **Google OAuth** — their Google `sub` ID is used as a persistent `user_id`
- Documents are stored **once per unique file** (identified by MD5 hash) in MongoDB
- Each chunk tracks a `session_ids` array — if two users upload the same PDF, they both point to the same stored chunks
- Retrieval is always **pre-filtered by `user_id`** so users only see their own documents
- Chat history is stored per `user_id` in a separate MongoDB collection

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Haiku (Anthropic) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Vector Database | MongoDB Atlas Vector Search |
| RAG Orchestration | LangGraph |
| Authentication | Google OAuth 2.0 (`streamlit-oauth`) |
| Chat Persistence | MongoDB (`chat_sessions` collection) |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |

---

## Setup

### Prerequisites

- Python 3.11
- MongoDB Atlas account (free tier works)
- Anthropic API key
- Google Cloud project with OAuth 2.0 credentials

### 1. Clone the repo

```bash
git clone https://github.com/sharanya39/dochat-ai.git
cd dochat-ai
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0
DB_NAME=dochat_db
COLLECTION_NAME=dochat_vectors
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
REDIRECT_URI=http://localhost:8501
```

### 5. Set up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → APIs & Services → Credentials → Create OAuth 2.0 Client ID
3. Application type: **Web application**
4. Add to **Authorized redirect URIs**:
   - `http://localhost:8501` (local dev)
   - `https://your-app.streamlit.app` (production)
5. Copy the Client ID and Client Secret to your `.env`

### 6. Set up MongoDB Atlas Vector Search Index

In MongoDB Atlas, create a vector search index on your `dochat_vectors` collection with the name `default`:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "session_ids"
    }
  ]
}
```

### 7. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, sign in with Google, and start uploading PDFs.

---

## Deploying on Streamlit Cloud

### 1. Push code to GitHub

```bash
git add .
git commit -m "Deploy DocChat AI"
git push origin main
```

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Create app"** → select your GitHub repo
3. Set **Main file path** to `app.py`

### 3. Add secrets

In **Settings → Secrets**, paste:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0"
DB_NAME = "dochat_db"
COLLECTION_NAME = "dochat_vectors"
GOOGLE_CLIENT_ID = "your_google_client_id"
GOOGLE_CLIENT_SECRET = "your_google_client_secret"
REDIRECT_URI = "https://your-app.streamlit.app"
```

### 4. Allow all IPs in MongoDB Atlas

Streamlit Cloud uses dynamic IPs:

1. MongoDB Atlas → **Network Access** → **Add IP Address**
2. Enter `0.0.0.0/0` → **Confirm**

### 5. Deploy

Click **"Deploy"** — your app will be live.

---

## Project Structure

```
dochat-ai/
├── app.py              # Streamlit UI + Google OAuth
├── rag_engine.py       # Agentic RAG pipeline (LangGraph)
├── chat_history.py     # MongoDB chat persistence
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python 3.11 pin for Streamlit Cloud
├── .env                # Environment variables (not committed)
└── .gitignore
```

---

## Contributing

Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Acknowledgements

- Developed by [Sharanya](https://github.com/sharanya39) with AI assistance from [Claude](https://claude.ai) by Anthropic
- Powered by [LangChain](https://langchain.com), [LangGraph](https://langchain-ai.github.io/langgraph/), and [MongoDB Atlas](https://www.mongodb.com/atlas)

> This project was built as a portfolio project exploring Agentic RAG, LangGraph, Google OAuth, and Generative AI with guidance from Claude Code (Anthropic). The architecture, debugging, and deployment were all done collaboratively with Claude Code.
