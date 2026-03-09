import streamlit as st
import tempfile
import os
import threading
import time
import json
import sqlite3
import networkx as nx
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List
from flashrank import Ranker, RerankRequest

# --- Initial Setup & Observability ---
if "phoenix_session" not in st.session_state:
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    session = px.launch_app()
    st.session_state["phoenix_session"] = session

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_MODEL = "llama3:8b"
COLLECTION_NAME = "general_docs"
CHAT_MEMORY_COLLECTION = "chat_history"
SHARED_DOCS_DIR = "/app/docs"
KG_PATH = os.path.join(SHARED_DOCS_DIR, "knowledge_graph.graphml")

st.set_page_config(page_title="LocalRAG", layout="wide")


# Ensure Qdrant Collections Exist
def init_qdrant_collections():
    client = QdrantClient(url=QDRANT_URL)
    for col in [COLLECTION_NAME, CHAT_MEMORY_COLLECTION]:
        if not client.collection_exists(col):
            client.create_collection(
                collection_name=col,
                vectors_config=rest.VectorParams(size=4096, distance=rest.Distance.COSINE)
            )


init_qdrant_collections()

# Initialize Knowledge Graph
if not os.path.exists(KG_PATH):
    G = nx.DiGraph()
    nx.write_graphml(G, KG_PATH)


# --- File Loading Helpers ---
def load_local_file(file_path):
    ext = file_path.lower()
    try:
        if ext.endswith(".pdf"): return PyMuPDFLoader(file_path).load()
        if ext.endswith(".docx"): return Docx2txtLoader(file_path).load()
        # Fallback to standard TextLoader for MD and TXT to prevent silent dependency crashes
        return TextLoader(file_path, encoding="utf-8").load()
    except Exception as e:
        print(f"File Load Error for {file_path}: {e}")
        return []

# --- Background Sync Service (Watchdog) ---
class DocumentSyncHandler(FileSystemEventHandler):
    def delete_file_context(self, file_path):
        try:
            client = QdrantClient(url=QDRANT_URL)
            if client.collection_exists(COLLECTION_NAME):
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=rest.Filter(
                        must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=file_path))]
                    ),
                )
        except Exception as e:
            pass

    def process_file(self, file_path):
        filename = os.path.basename(file_path).lower()
        if filename in {"app.py", "docker-compose.yml", "requirements.txt", "makefile",
                        "dockerfile"} or filename.endswith((".py", ".yml", ".graphml")):
            return
        if not filename.endswith((".pdf", ".docx", ".md", ".txt")):
            return

        self.delete_file_context(file_path)

        if os.path.exists(file_path):
            try:
                docs = load_local_file(file_path)
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
                QdrantVectorStore.from_documents(splits, embeddings, url=QDRANT_URL, collection_name=COLLECTION_NAME)
            except Exception as e:
                print(f"Sync Error (Index): {e}")

    def on_created(self, event):
        if not event.is_directory: self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory: self.process_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory: self.delete_file_context(event.src_path)


@st.cache_resource
def start_watchdog():
    if not os.path.exists(SHARED_DOCS_DIR): os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
    observer = PollingObserver()
    observer.schedule(DocumentSyncHandler(), SHARED_DOCS_DIR, recursive=True)
    threading.Thread(target=observer.start, daemon=True).start()
    return observer


start_watchdog()


# --- Agent State & Graph Nodes ---
class AgentState(TypedDict):
    question: str
    chat_history: List[dict]
    search_queries: List[str]
    documents: List[dict]
    generation: str
    hallucination_count: int
    user_profile: List[str]  # Dynamic facts array
    kg_context: str  # GraphRAG context


def transform_query(state):
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    history = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"][-3:]])
    prompt = f"""You are a query optimization assistant. 
    Analyze the recent conversation and the user's latest question. 
    Generate 3 distinct search queries to maximize retrieval from a vector database.
    Return ONLY the 3 queries separated by newlines. No formatting.

    Recent Chat: {history}
    Latest Question: {state['question']}"""

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    if state["question"] not in queries:
        queries.append(state["question"])

    return {"search_queries": queries}


def retrieve(state):
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # 1. Retrieve Semantic Memory (Past Conversations)
    chat_store = QdrantVectorStore(client=client, collection_name=CHAT_MEMORY_COLLECTION, embedding=embeddings)
    chat_retriever = chat_store.as_retriever(search_kwargs={"k": 2})
    past_interactions = chat_retriever.invoke(state["question"])

    # 2. GraphRAG Retrieval (NetworkX)
    kg_context = ""
    try:
        G = nx.read_graphml(KG_PATH)
        words = state["question"].lower().split()
        subgraph_nodes = [n for n in G.nodes if any(w in n.lower() for w in words)]
        kg_edges = []
        for n in subgraph_nodes:
            kg_edges.extend(G.edges(n, data=True))
        if kg_edges:
            kg_context = "\n".join(
                [f"{u} -> {data.get('relation', 'is related to')} -> {v}" for u, v, data in kg_edges[:5]])
    except Exception:
        pass

    # 3. Standard Document Retrieval
    selected_contexts = st.session_state.get("selected_context", [SHARED_DOCS_DIR])
    search_filter = None
    if SHARED_DOCS_DIR not in selected_contexts:
        valid_files = []
        for path in selected_contexts:
            if os.path.isfile(path):
                valid_files.append(path)
            else:
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith((".pdf", ".docx", ".md", ".txt")):
                            valid_files.append(os.path.join(root, f))
        valid_files = list(set(valid_files))
        if valid_files:
            search_filter = rest.Filter(
                must=[rest.FieldCondition(key="metadata.source", match=rest.MatchAny(any=valid_files))])

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": search_filter})

    unique_docs = {}
    for q in state["search_queries"]:
        for d in retriever.invoke(q):
            if d.page_content not in unique_docs:
                unique_docs[d.page_content] = d

    # Add semantic memory into the re-ranking pool
    for d in past_interactions:
        d.metadata["source"] = "Past Conversation Memory"
        unique_docs[d.page_content] = d

    aggregated_docs = list(unique_docs.values())

    if not aggregated_docs:
        return {"documents": [], "hallucination_count": 0, "kg_context": kg_context}

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./ranker_cache")
    rerankrequest = RerankRequest(
        query=state["question"],
        passages=[{"text": d.page_content, "meta": d.metadata} for d in aggregated_docs]
    )

    ranked = ranker.rerank(rerankrequest)[:4]
    top_docs = [{"text": r["text"], "source": r["meta"]["source"]} for r in ranked]

    return {"documents": top_docs, "hallucination_count": 0, "kg_context": kg_context}


def generate(state):
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

    context_str = ""
    if state["documents"]:
        for d in state["documents"]:
            context_str += f"Source: {d['source']}\nContent: {d['text']}\n\n"

    # Inject Dynamic Profile & GraphRAG Data
    profile_str = "\n".join(state.get("user_profile", []))
    kg_str = state.get("kg_context", "")

    history = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"][-5:]])

    instruction = f"""You are an intelligent research assistant. 
    User Profile Facts: {profile_str if profile_str else 'None established yet.'}
    Knowledge Graph Context: {kg_str if kg_str else 'None.'}

    Answer the question using the Conversation History and the Context provided.
    CRITICAL RULE: If you use information from the Context, you MUST cite the 'Source' inline where the claim is made. 
    If the Context is missing or insufficient, use your internal knowledge."""

    if state["hallucination_count"] > 0:
        instruction += "\nIMPORTANT: Your previous answer failed validation. Ground your answer strictly in the context."

    prompt = f"{instruction}\n\nConversation History:\n{history}\n\nContext:\n{context_str}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"generation": response.content}


def update_retry(state):
    return {"hallucination_count": state["hallucination_count"] + 1}


def grade_hallucination(state):
    if state["hallucination_count"] >= 3:
        return "update_memory"

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    context = "\n\n".join([d["text"] for d in state["documents"]]) if state["documents"] else ""

    prompt = f"""Rules:
    1. If context exists and answer contradicts it, score 'no'.
    2. If context is empty, and answer uses general knowledge, score 'yes'.
    3. Return JSON: {{"score": "yes"}} or {{"score": "no"}}.
    Context: {context}
    Answer: {state["generation"]}"""

    try:
        response = llm.invoke(prompt)
        score = json.loads(response.content).get("score", "yes")
    except:
        score = "yes"

    return "update_memory" if score == "yes" else "rewrite"


def update_memory(state):
    """Semantic Memory, Fact Extraction, and GraphRAG Integration."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")

    # 1. Update Semantic Memory (Qdrant)
    interaction = f"User: {state['question']}\nAgent: {state['generation']}"
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    store = QdrantVectorStore(client=QdrantClient(url=QDRANT_URL), collection_name=CHAT_MEMORY_COLLECTION,
                              embedding=embeddings)
    store.add_texts([interaction], metadatas=[{"source": "Semantic Memory"}])

    # 2. Extract Dynamic Facts & Graph Triples
    prompt = f"""Extract permanent facts and knowledge graph triples from this interaction.
    Interaction: User asked '{state['question']}'. Agent replied '{state['generation']}'.
    Return JSON format: 
    {{"facts": ["User likes X", "User is building Y"], "triples": [["Subject", "Predicate", "Object"]]}}
    If nothing notable, return empty lists."""

    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)

        # Update Facts
        new_facts = state.get("user_profile", []) + data.get("facts", [])

        # Update GraphRAG (NetworkX)
        if data.get("triples"):
            G = nx.read_graphml(KG_PATH) if os.path.exists(KG_PATH) else nx.DiGraph()
            for subj, pred, obj in data.get("triples", []):
                G.add_node(subj)
                G.add_node(obj)
                G.add_edge(subj, obj, relation=pred)
            nx.write_graphml(G, KG_PATH)

        return {"user_profile": list(set(new_facts))}
    except Exception as e:
        return {"user_profile": state.get("user_profile", [])}


# --- Graph Construction (With Checkpointing) ---
# Checkpointer for time-travel and thread persistence
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

workflow = StateGraph(AgentState)
workflow.add_node("transform_query", transform_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("update_retry", update_retry)
workflow.add_node("update_memory", update_memory)

workflow.set_entry_point("transform_query")
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("update_retry", "generate")
workflow.add_conditional_edges("generate", grade_hallucination,
                               {"update_memory": "update_memory", "rewrite": "update_retry"})
workflow.add_edge("update_memory", END)

# Compile with the Checkpointer
app_graph = workflow.compile(checkpointer=memory)


# --- Streamlit Chat UI ---
def get_paths(root):
    paths = [root]
    for r, dirs, files in os.walk(root):
        for d in dirs: paths.append(os.path.join(r, d))
        for f in files:
            if f.lower().endswith((".pdf", ".docx", ".md", ".txt")): paths.append(os.path.join(r, f))
    return paths


st.title("LocalRAG Agent")
st.caption("Sqlite Checkpointing | Semantic Recall | Fact Extraction | GraphRAG")

# Initialize Thread ID for Checkpointer
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_1"  # In a real app, generate UUID per user

config = {"configurable": {"thread_id": st.session_state.thread_id}}

with st.sidebar:
    st.header("🎯 Context Filter")
    all_paths = get_paths(SHARED_DOCS_DIR)

    selected = st.multiselect(
        "Focus search on (Select multiple):",
        options=all_paths,
        format_func=lambda x: x.replace(SHARED_DOCS_DIR, "ROOT"),
        help="Leave blank to search everything, or pick specific folders/files."
    )

    st.session_state["selected_context"] = selected if selected else [SHARED_DOCS_DIR]

    if selected:
        with st.spinner("Verifying & Auto-Pushing Context..."):
            client = QdrantClient(url=QDRANT_URL)
            handler = DocumentSyncHandler()

            files_to_check = []
            for p in selected:
                if os.path.isfile(p):
                    files_to_check.append(p)
                else:
                    for r, _, files in os.walk(p):
                        for f in files:
                            if f.lower().endswith((".pdf", ".docx", ".md", ".txt")):
                                files_to_check.append(os.path.join(r, f))

            for fp in set(files_to_check):
                records, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=rest.Filter(
                        must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=fp))]),
                    limit=1
                )
                if not records: handler.process_file(fp)

    st.markdown("---")
    st.header("🧠 Agent Memory State")
    current_state = app_graph.get_state(config)
    if current_state.values:
        st.write("**Extracted Facts:**")
        for fact in current_state.values.get("user_profile", []):
            st.caption(f"- {fact}")

    st.markdown("---")
    st.header("🗄️ Knowledge Base")
    st.link_button("📂 Open File Manager", "http://localhost:8080", use_container_width=True)

    # NEW: Live Database Status
    try:
        col_info = QdrantClient(url=QDRANT_URL).get_collection(COLLECTION_NAME)
        st.success(f"📊 Vector DB contains **{col_info.points_count}** chunks.")
    except:
        st.warning("📊 Vector DB is currently empty.")

    if st.button("🗑️ Reset All Databases", use_container_width=True):
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(COLLECTION_NAME): client.delete_collection(COLLECTION_NAME)
        if client.collection_exists(CHAT_MEMORY_COLLECTION): client.delete_collection(CHAT_MEMORY_COLLECTION)
        if os.path.exists(KG_PATH): os.remove(KG_PATH)
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent routing, retrieving, verifying, and updating memory..."):
            # Pass the Checkpointer Config into the invoke call
            final_state = app_graph.invoke({
                "question": prompt,
                "chat_history": st.session_state.messages[:-1],
                "hallucination_count": 0
            }, config=config)


        def stream_text():
            for word in final_state["generation"].split(" "):
                yield word + " "
                time.sleep(0.015)


        st.write_stream(stream_text)

        with st.expander("Debug Trace"):
            st.write(f"**Extracted GraphRAG Context:** {final_state.get('kg_context', 'None')}")
            st.write(f"**Expanded Queries:** {final_state['search_queries']}")
            st.write(f"**Chunks Retrieved & Ranked:** {len(final_state['documents'])}")
            st.write(f"**Correction Loops:** {final_state['hallucination_count']}")

    st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})