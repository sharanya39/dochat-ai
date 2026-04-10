# DocChat AI

> **Chat with your documents using Agentic AI** — Upload any PDF and get accurate, cited answers powered by Claude AI, MongoDB Atlas Vector Search, and LangGraph.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-red)](https://langchain-ai.github.io/langgraph/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas_Vector_Search-green)](https://mongodb.com/atlas)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b)](https://streamlit.io)

---

## Why DocChat AI?

Tired of scrolling through long PDFs to find answers? DocChat AI lets you **ask questions in plain English** and get precise answers with page citations — instantly.

- No more manual searching through documents
- Answers grounded in your actual document — no hallucinations
- Intelligent re-retrieval if the first search isn't good enough
- Each user's documents are fully private and isolated

## Demo

Upload any PDF → Ask questions → Get accurate, cited answers grounded in your document.

## Contributing

Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## Acknowledgements

- Developed by [Sharanya](https://github.com/sharanya39) with AI assistance from [Claude](https://claude.ai) by Anthropic
- Powered by [LangChain](https://langchain.com), [LangGraph](https://langchain-ai.github.io/langgraph/), and [MongoDB Atlas](https://www.mongodb.com/atlas)

> This project was built as a portfolio project exploring Agentic RAG, LangGraph, and Generative AI with guidance from Claude (Anthropic). The architecture, debugging, and deployment were all done collaboratively with Claude Code.

## Architecture

```
PDF Upload → Chunking → Embedding → MongoDB Atlas
                                          ↓
User Question → Retrieve → Evaluate → [Re-retrieve if needed] → Generate Answer
```

**Agentic RAG Pipeline (LangGraph):**
1. **Retrieve** — Fetch top-5 relevant chunks via vector search
2. **Evaluate** — Claude scores retrieval relevance (1-10)
3. **Expand & Re-retrieve** — If score < 6, generate better queries and re-retrieve
4. **Generate** — Claude synthesizes final answer with source citations

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Haiku (Anthropic) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Vector Database | MongoDB Atlas Vector Search |
| RAG Orchestration | LangGraph |
| LangChain Integrations | langchain-anthropic, langchain-mongodb, langchain-huggingface |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |

## Features

- Upload multiple PDF documents
- Per-session document isolation
- Duplicate ingestion prevention — same PDF won't be re-ingested
- Agentic re-retrieval when initial results are weak
- In-memory LLM cache for consistent, fast repeated answers
- Source citations with page numbers

## Setup

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (free tier works)
- Anthropic API key

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
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=dochat_db
COLLECTION_NAME=dochat_vectors
```

### 5. Set up MongoDB Atlas Vector Search Index

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
      "path": "session_id"
    }
  ]
}
```

### 6. Run the app

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (`http://localhost:8501`)
2. Upload one or more PDF files in the sidebar
3. Ask questions in the chat input
4. Get answers with source page citations

## Deploying on Streamlit Cloud

### 1. Push code to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dochat-ai.git
git push -u origin main
```

### 2. Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Create app"**
3. Connect your GitHub account and select the `dochat-ai` repository
4. Set **Main file path** to `app.py`
5. Click **"Advanced settings"**

### 3. Add secrets
In **Advanced settings → Secrets**, paste your credentials in TOML format:
```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0"
DB_NAME = "dochat_db"
COLLECTION_NAME = "dochat_vectors"
```

### 4. Allow all IPs in MongoDB Atlas
Streamlit Cloud uses dynamic IPs, so you must whitelist all IPs:
1. Go to MongoDB Atlas → **Network Access**
2. Click **"Add IP Address"**
3. Enter `0.0.0.0/0` → click **Confirm**

### 5. Deploy
Click **"Deploy"** — the app will be live at:
`https://YOUR_USERNAME-dochat-ai-app-xxxx.streamlit.app`

---

## Project Structure

```
dochat-ai/
├── app.py              # Streamlit UI
├── rag_engine.py       # Agentic RAG pipeline (LangGraph)
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version pin (3.11) for Streamlit Cloud
├── .env                # Environment variables (not committed)
└── .gitignore
```
