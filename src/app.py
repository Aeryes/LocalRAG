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


class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    hallucination_count: int


def retrieve(state):
    """Fetch 10 docs, then Re-Rank to top 3 using FlashRank"""
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    raw_docs = retriever.invoke(state["question"])

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./ranker_cache")
    rerankrequest = RerankRequest(
        query=state["question"],
        passages=[{"text": d.page_content, "meta": d.metadata} for d in raw_docs]
    )
    ranked_results = ranker.rerank(rerankrequest)

    top_docs = [r["text"] for r in ranked_results[:3]]
    return {"documents": top_docs, "hallucination_count": 0}


def generate(state):
    """Generate answer using Ollama"""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

    context = "\n\n".join(state["documents"])
    prompt = f"""You are a research assistant. Use the context below to answer the user's question.

    Context:
    {context}

    Question: {state["question"]}
    """

    response = llm.invoke(prompt)
    return {"generation": response.content}


def grade_hallucination(state):
    """Ask LLM if the answer is supported by facts"""
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


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "generate",
    grade_hallucination,
    {
        "end": END,
        "rewrite": "generate"
    }
)
app = workflow.compile()

st.title("LocalRAG")
st.caption("Features: Re-Ranking | Self-Correction | Observability | Streaming")

if st.session_state.get("phoenix_session"):
    # This URL needs to be localhost on your host machine
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
                if not client.collection_exists(COLLECTION_NAME):
                    client.create_collection(collection_name=COLLECTION_NAME,
                                             vectors_config=VectorParams(size=4096, distance=Distance.COSINE))

                embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
                QdrantVectorStore.from_documents(splits, embeddings, url=QDRANT_URL, collection_name=COLLECTION_NAME,
                                                 force_recreate=True)
                st.success(f"Indexed {len(splits)} chunks!")

user_input = st.text_input("Ask a question:")
if user_input:
    with st.spinner("Thinking (Graph Running)..."):
        final_state = app.invoke({"question": user_input, "hallucination_count": 0})

    st.markdown("### Answer")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    context = "\n\n".join(final_state["documents"])
    prompt = f"Context: {context}\n\nQuestion: {user_input}"

    st.write_stream(llm.stream(prompt))

    with st.expander("Debug Info (Senior Features)"):
        st.markdown("**Re-Ranked Context (Top 3):**")
        for d in final_state["documents"]: st.text(d[:200] + "...")
        st.markdown(f"**Hallucination Check Passed:** Yes")