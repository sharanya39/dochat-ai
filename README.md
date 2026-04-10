# DocChat AI

A document Q&A application powered by Agentic RAG (Retrieval-Augmented Generation). Upload PDF documents and chat with them using Claude AI, MongoDB Atlas Vector Search, and LangGraph.

## Demo

Upload any PDF → Ask questions → Get accurate, cited answers grounded in your document.

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

## Project Structure

```
dochat-ai/
├── app.py              # Streamlit UI
├── rag_engine.py       # Agentic RAG pipeline (LangGraph)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
└── .gitignore
```
