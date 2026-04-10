import os
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pymongo import MongoClient
import langgraph.graph as lg
from langgraph.graph import StateGraph, END

load_dotenv()

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# ── Models ────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",   # fast + cheap for demos
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0,  # 0 = deterministic, consistent answers
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── MongoDB ───────────────────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["dochat_db"]["dochat_vectors"]


def get_vector_store(session_id: str) -> MongoDBAtlasVectorSearch:
    """Each user session gets an isolated namespace via metadata filtering."""
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="default",
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )


# ── Document Ingestion ────────────────────────────────────────────────────────
def ingest_document(file_path: str, session_id: str, original_filename: str = None) -> int:
    """Load PDF, chunk it, embed and store in MongoDB with session tag."""
    filename = original_filename or os.path.basename(file_path)
    existing = collection.count_documents({"source": filename})
    if existing > 0:
        return existing  # already ingested, skip

    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with session_id and real filename
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
        chunk.metadata["source"] = filename

    vector_store = get_vector_store(session_id)
    vector_store.add_documents(chunks)
    return len(chunks)


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
    """Retrieve top-k docs for the current question."""
    vector_store = get_vector_store(state["session_id"])
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    docs = retriever.invoke(state["question"])
    print(f"[DEBUG] Question: {state['question']}")
    print(f"[DEBUG] Retrieved {len(docs)} docs")
    for i, doc in enumerate(docs):
        print(f"[DEBUG] Doc {i}: {doc.page_content[:100]}")
    return {**state, "retrieved_docs": docs}


def evaluate_retrieval_node(state: RAGState) -> RAGState:
    """Check if retrieved docs are relevant. If not, expand query."""
    if not state["retrieved_docs"]:
        return {**state, "retrieved_docs": [], "answer": ""}

    # Ask Claude to judge relevance (lightweight check)
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
        import json
        result = chain.invoke({"question": state["question"], "context": context_text})
        # Extract JSON from response
        start = result.find("{")
        end = result.rfind("}") + 1
        parsed = json.loads(result[start:end])
        score = parsed.get("score", 7)
        queries = parsed.get("queries", [])

        if score < 6 and state["retry_count"] < 2 and queries:
            return {**state, "expanded_queries": queries}
    except Exception:
        pass  # If parsing fails, proceed with what we have

    return {**state, "expanded_queries": []}


def expand_and_retrieve_node(state: RAGState) -> RAGState:
    """Re-retrieve using expanded queries and merge results."""
    vector_store = get_vector_store(state["session_id"])
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
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
        "retrieved_docs": all_docs[:8],  # cap at 8 chunks
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
- If not found, say "This information is not in the uploaded document"
- Be concise but complete"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": state["chat_history"][-6:],  # last 3 turns
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
    """Delete all vectors for a given session."""
    collection.delete_many({"session_id": session_id})
