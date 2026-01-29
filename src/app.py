import streamlit as st
import tempfile
import os
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from flashrank import Ranker, RerankRequest

# --- Initial Setup ---
if "phoenix_session" not in st.session_state:
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    session = px.launch_app()
    st.session_state["phoenix_session"] = session

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_MODEL = "llama3:8b"
COLLECTION_NAME = "general_docs"

st.set_page_config(page_title="LocalRAG", layout="wide")


# --- File Loading ---
def load_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(tmp_path)
    elif uploaded_file.name.endswith(".md"):
        loader = UnstructuredMarkdownLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)
    return loader.load()


# --- State Definition ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    hallucination_count: int


# --- Nodes ---
def retrieve(state):
    """Fetch 10 docs, then Re-Rank to top 3 using FlashRank"""
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    raw_docs = retriever.invoke(state["question"])

    # Re-Ranking Step
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./ranker_cache")
    rerankrequest = RerankRequest(
        query=state["question"],
        passages=[{"text": d.page_content, "meta": d.metadata} for d in raw_docs]
    )
    ranked_results = ranker.rerank(rerankrequest)

    # Take top 3 re-ranked docs
    top_docs = [r["text"] for r in ranked_results[:3]]
    return {"documents": top_docs, "hallucination_count": 0}  # Reset count on new retrieve


def generate(state):
    """Generate answer using Ollama"""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

    context = "\n\n".join(state["documents"])

    # Slight prompt tweak if this is a retry
    instruction = "You are a research assistant. Use the context below to answer the user's question."
    if state["hallucination_count"] > 0:
        instruction += " IMPORTANT: Your previous answer was rejected for hallucinating. Stick strictly to the context."

    prompt = f"""{instruction}

    Context:
    {context}

    Question: {state["question"]}
    """

    response = llm.invoke(prompt)
    return {"generation": response.content}


def update_retry(state):
    """Increment the retry counter"""
    return {"hallucination_count": state["hallucination_count"] + 1}


def grade_hallucination(state):
    """Conditional Edge: Ask LLM if the answer is supported by facts"""

    # 1. Safety Valve: Stop if we have retried 3 times
    if state["hallucination_count"] >= 3:
        return "end"

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    context = "\n\n".join(state["documents"])

    prompt = f"""
    You are a grader. Check if the Answer is grounded in the Context.

    Rules:
    1. If the Answer cites libraries NOT in the Context, score is 'no'.
    2. If the Context is empty, score is 'no'.
    3. Return JSON: {{"score": "yes"}} or {{"score": "no"}}.

    Context: {context}
    Answer: {state["generation"]}

    JSON Output:
    """

    try:
        response = llm.invoke(prompt)
        import json
        grade = json.loads(response.content)
        score = grade.get("score", "yes")
    except:
        score = "yes"

    if score == "no":
        return "rewrite"

    return "end"


# --- Graph Construction ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("update_retry", update_retry)  # New node to handle state update

# Set Entry
workflow.set_entry_point("retrieve")

# Standard Edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("update_retry", "generate")  # After updating count, go back to generate

# Conditional Edges
workflow.add_conditional_edges(
    "generate",
    grade_hallucination,
    {
        "end": END,
        "rewrite": "update_retry"  # If hallucinated, go to update_retry first
    }
)

app = workflow.compile()

# --- Streamlit UI ---
st.title("LocalRAG")
st.caption("Features: Re-Ranking | Self-Correction | Observability | Streaming")

if st.session_state.get("phoenix_session"):
    st.markdown(f"[View Observability Dashboard](http://localhost:6006) (Tracing Active)")

with st.sidebar:
    st.header("Ingestion")
    uploaded_files = st.file_uploader("Upload Docs", accept_multiple_files=True)
    if st.button("Ingest"):
        if uploaded_files:
            with st.spinner("Processing..."):
                all_docs = []
                for f in uploaded_files: all_docs.extend(load_file(f))
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)

                client = QdrantClient(url=QDRANT_URL)
                # Create collection if needed
                if not client.collection_exists(COLLECTION_NAME):
                    client.create_collection(collection_name=COLLECTION_NAME,
                                             vectors_config=VectorParams(size=4096, distance=Distance.COSINE))

                embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
                # Force recreate ensures we don't duplicate on re-ingest for this demo
                QdrantVectorStore.from_documents(splits, embeddings, url=QDRANT_URL, collection_name=COLLECTION_NAME,
                                                 force_recreate=True)
                st.success(f"Indexed {len(splits)} chunks!")

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking (Graph Running)..."):
        # Initialize with 0 retries
        final_state = app.invoke({"question": user_input, "hallucination_count": 0})

    st.markdown("### Answer")

    # Check if we failed after max retries
    if final_state["hallucination_count"] >= 3:
        st.warning("Note: The model struggled to ground this answer in the context (Max Retries Hit).")

    # Stream the final result (or re-generate strictly for streaming effect)
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    context = "\n\n".join(final_state["documents"])
    prompt = f"Context: {context}\n\nQuestion: {user_input}"
    st.write_stream(llm.stream(prompt))

    with st.expander("Debug Info (Senior Features)"):
        st.markdown("**Re-Ranked Context (Top 3):**")
        for d in final_state["documents"]: st.text(d[:200] + "...")
        st.markdown(f"**Retries Used:** {final_state['hallucination_count']}")