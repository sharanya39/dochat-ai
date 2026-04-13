import os
import json
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pymongo import MongoClient
from langgraph.graph import StateGraph, END

load_dotenv()

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# ── Models ────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── MongoDB ───────────────────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["dochat_db"]["dochat_vectors"]


def get_vector_store() -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="default",
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )


# ── Document Ingestion ────────────────────────────────────────────────────────
def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file content to detect truly identical files."""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ingest_document(file_path: str, session_id: str, original_filename: str = None) -> tuple[int, bool]:
    """
    Shared Knowledge Base ingestion using filename + content hash:
    - Same filename + same content → fetch from DB, register session
    - Same filename + different content → treat as new doc (different hash)
    - New doc → ingest fresh
    Returns (chunk_count, already_existed)
    """
    filename = original_filename or os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    # Remove session_id from any chunks with same filename but different content
    # Handles: same filename, different content re-upload by same session
    collection.update_many(
        {"source": filename, "file_hash": {"$ne": file_hash}, "session_ids": session_id},
        {"$pull": {"session_ids": session_id}}
    )
    # Clean up orphaned chunks (no sessions left after removal)
    collection.delete_many({"source": filename, "file_hash": {"$ne": file_hash}, "session_ids": {"$size": 0}})

    # Check if exact same content already exists in DB
    existing_count = collection.count_documents({"source": filename, "file_hash": file_hash})

    if existing_count > 0:
        # Same file content — just register this session_id
        collection.update_many(
            {"source": filename, "file_hash": file_hash, "session_ids": {"$nin": [session_id]}},
            {"$addToSet": {"session_ids": session_id}}
        )
        return existing_count, True  # already existed

    # New document or same name but different content — ingest fresh
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with filename, hash and session_ids list
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["session_ids"] = [session_id]

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks), False  # newly ingested


# ── Agentic RAG State ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    session_id: str
    chat_history: List
    retrieved_docs: List[Document]
    answer: str
    retry_count: int
    expanded_queries: List[str]


# ── Agentic Nodes ─────────────────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve top-k docs filtering by session_ids array."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "pre_filter": {"session_ids": {"$in": [state["session_id"]]}},
        },
    )
    docs = retriever.invoke(state["question"])
    return {**state, "retrieved_docs": docs}


def evaluate_retrieval_node(state: RAGState) -> RAGState:
    """Check if retrieved docs are relevant. If not, expand query."""
    if not state["retrieved_docs"]:
        return {**state, "retrieved_docs": [], "answer": ""}

    eval_prompt = ChatPromptTemplate.from_template(
        """Given the question: "{question}"

        And these retrieved document excerpts:
        {context}

        Rate relevance 1-10. If < 6, suggest 2 better search queries.
        Reply ONLY as JSON: {{"score": N, "queries": ["q1", "q2"]}}"""
    )
    context_text = "\n---\n".join([d.page_content[:200] for d in state["retrieved_docs"][:3]])

    chain = eval_prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": state["question"], "context": context_text})
        start = result.find("{")
        end = result.rfind("}") + 1
        parsed = json.loads(result[start:end])
        score = parsed.get("score", 7)
        queries = parsed.get("queries", [])

        if score < 6 and state["retry_count"] < 2 and queries:
            return {**state, "expanded_queries": queries}
    except Exception:
        pass

    return {**state, "expanded_queries": []}


def expand_and_retrieve_node(state: RAGState) -> RAGState:
    """Re-retrieve using expanded queries and merge results."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "pre_filter": {"session_ids": {"$in": [state["session_id"]]}},
        },
    )
    all_docs = list(state["retrieved_docs"])
    seen = {d.page_content for d in all_docs}

    for query in state["expanded_queries"][:2]:
        new_docs = retriever.invoke(query)
        for doc in new_docs:
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)

    return {
        **state,
        "retrieved_docs": all_docs[:8],
        "retry_count": state["retry_count"] + 1,
        "expanded_queries": [],
    }


def generate_answer_node(state: RAGState) -> RAGState:
    """Generate final answer with citations using retrieved context."""
    if not state["retrieved_docs"]:
        return {**state, "answer": "No relevant documents found. Please upload a document first."}

    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'document')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in state["retrieved_docs"]
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful document assistant. Answer ONLY based on the provided context.

Context from uploaded documents:
{context}

Rules:
- If the answer is in the context, answer clearly and cite the source page
- If not found, say ONLY "This information is not in the uploaded document." Do NOT elaborate, do NOT describe what the document is about, do NOT use your general knowledge
- Be concise but complete"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": state["chat_history"][-6:],
        "question": state["question"],
    })

    return {**state, "answer": answer}


# ── Routing Logic ─────────────────────────────────────────────────────────────
def should_expand(state: RAGState) -> str:
    if state["expanded_queries"] and state["retry_count"] < 2:
        return "expand"
    return "generate"


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("expand_retrieve", expand_and_retrieve_node)
    graph.add_node("generate", generate_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_expand,
        {"expand": "expand_retrieve", "generate": "generate"},
    )
    graph.add_edge("expand_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_rag_graph()


def ask_question(question: str, session_id: str, chat_history: List) -> dict:
    """Main entry point — runs the agentic RAG pipeline."""
    initial_state = RAGState(
        question=question,
        session_id=session_id,
        chat_history=chat_history,
        retrieved_docs=[],
        answer="",
        retry_count=0,
        expanded_queries=[],
    )
    result = rag_graph.invoke(initial_state)

    # Deduplicate sources by page content
    seen = set()
    unique_sources = []
    for doc in result["retrieved_docs"]:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_sources.append(doc)

    return {
        "answer": result["answer"],
        "sources": [
            {
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "document"),
                "preview": doc.page_content[:150] + "...",
            }
            for doc in unique_sources[:3]
        ],
        "used_reretrieval": result["retry_count"] > 0,
    }


def clear_session_docs(session_id: str):
    """
    Remove session_id from all chunks.
    Only delete chunk if session_ids becomes empty (no users left).
    """
    # Remove this session from all chunks
    collection.update_many(
        {"session_ids": session_id},
        {"$pull": {"session_ids": session_id}}
    )
    # Delete chunks that no longer belong to any session
    collection.delete_many({"session_ids": {"$size": 0}})
