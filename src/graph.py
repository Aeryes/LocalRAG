import json
import logging
import os
import re
from typing import Any, Dict, List, TypedDict
from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from constants import (
    CHAT_MEMORY_COLLECTION,
    DENSE_TOP_K,
    DOC_COLLECTION_NAME,
    FINAL_TOP_K,
    GRAPH_TOP_K,
    LEXICAL_TOP_K,
    MAX_CHAT_HISTORY_FOR_ANSWER,
    MAX_CONTEXT_CHARS,
    MAX_HALLUCINATION_RETRIES,
    MAX_QUERY_HISTORY,
    MEMORY_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    QDRANT_URL,
    RERANK_TOP_K,
    SHARED_DOCS_DIR,
    VISUAL_TOP_K,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    ONTOLOGY,
)
from neo4j import GraphDatabase
from tools import get_tools
from multimodal import visual_search
from services import collect_supported_files, lexical_search

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    chat_history: List[dict]
    query_plan: Dict[str, Any]
    search_queries: List[str]
    documents: List[dict]
    generation: str
    answer_payload: Dict[str, Any]
    hallucination_count: int
    user_profile: List[dict]
    kg_context: str
    selected_context: List[str]
    retrieval_trace: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]


@lru_cache(maxsize=1)
def get_ranker():
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./ranker_cache")

@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_chat_llm(*, temperature: float = 0.2, as_json: bool = False):
    kwargs: Dict[str, Any] = {
        "model": OLLAMA_CHAT_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "temperature": temperature,
        "mirostat": None,
        "mirostat_eta": None,
        "mirostat_tau": None,
        "tfs_z": None,
    }
    if as_json:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def normalize_fact_records(facts: List[Any]) -> List[Dict[str, Any]]:
    normalized = []
    seen = set()
    for fact in facts or []:
        if isinstance(fact, dict):
            text = str(fact.get("text", "")).strip()
            confidence = float(fact.get("confidence", 0.7))
        else:
            text = str(fact).strip()
            confidence = 0.7
        if text and text.lower() not in seen:
            seen.add(text.lower())
            normalized.append({"text": text, "confidence": confidence})
    return normalized


def fact_texts_for_prompt(facts: List[Any]) -> str:
    records = normalize_fact_records(facts)
    if not records:
        return "None established yet."
    return "\n".join([f"- {item['text']} (confidence={item['confidence']:.2f})" for item in records])


def plan_query(state: AgentState):
    llm = get_chat_llm(temperature=0.1, as_json=True)
    history = "\n".join(
        [f"{message['role']}: {message['content']}" for message in state.get("chat_history", [])[-MAX_QUERY_HISTORY:]]
    )

    prompt = f"""You are a retrieval planner for a local multimodal RAG system.
Given the user's question and recent chat history, produce a compact JSON plan.

Rules:
- Always include the original question in search_queries.
- Produce 2 to 4 search queries total.
- retrieval_modes may include: ["dense", "lexical", "memory", "graph", "visual"].
- Prefer lexical for exact filenames, errors, config keys, code symbols, and APIs.
- Prefer visual if the question mentions charts, figures, images, screenshots, PDFs, tables, layouts, pages, diagrams, or asks "what does this page show".
- Use dense and lexical together for most document questions.

Return JSON only:
{{
  "intent": "lookup|compare|summarize|troubleshoot|design",
  "answer_style": "concise|normal|detailed",
  "search_queries": ["..."],
  "retrieval_modes": ["dense", "lexical", "memory", "graph", "visual"]
}}

Chat History:
{history}

Question:
{state['question']}
"""
    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        search_queries = [query.strip() for query in data.get("search_queries", []) if str(query).strip()]
        if state["question"] not in search_queries:
            search_queries.append(state["question"])
        retrieval_modes = [
            mode for mode in data.get("retrieval_modes", [])
            if mode in {"dense", "lexical", "memory", "graph", "visual"}
        ]
        if not retrieval_modes:
            retrieval_modes = ["dense", "lexical", "memory", "graph", "visual"]
        plan = {
            "intent": data.get("intent", "lookup"),
            "answer_style": data.get("answer_style", "normal"),
            "search_queries": search_queries[:4],
            "retrieval_modes": retrieval_modes,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse query plan JSON: {e}")
        question_lower = state["question"].lower()
        default_modes = ["dense", "lexical", "memory", "graph"]
        if any(token in question_lower for token in ["image", "table", "chart", "figure", "screenshot", "pdf", "page", "diagram", "visual"]):
            default_modes.append("visual")
        plan = {
            "intent": "lookup",
            "answer_style": "normal",
            "search_queries": [state["question"]],
            "retrieval_modes": default_modes,
        }
    except Exception as e:
        logger.error(f"Unexpected error in plan_query: {e}", exc_info=True)
        question_lower = state["question"].lower()
        default_modes = ["dense", "lexical", "memory", "graph"]
        if any(token in question_lower for token in ["image", "table", "chart", "figure", "screenshot", "pdf", "page", "diagram", "visual"]):
            default_modes.append("visual")
        plan = {
            "intent": "lookup",
            "answer_style": "normal",
            "search_queries": [state["question"]],
            "retrieval_modes": default_modes,
        }

    # Tool calling pass
    tool_calls = []
    tools = get_tools()
    if tools:
        llm_tools = get_chat_llm(temperature=0.1, as_json=False).bind_tools(tools)
        tool_prompt = f"Chat History:\n{history}\n\nQuestion:\n{state['question']}\n\nDo any of these tools help answer the question?"
        try:
            tool_response = llm_tools.invoke(tool_prompt)
            if hasattr(tool_response, "tool_calls"):
                tool_calls = tool_response.tool_calls
        except Exception as e:
            logger.error(f"Tool planning error: {e}", exc_info=True)

    return {"query_plan": plan, "search_queries": plan["search_queries"], "tool_calls": tool_calls}


def dense_search(query: str, selected_context: List[str], limit: int) -> List[Dict[str, Any]]:
    embeddings = get_embeddings()
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=DOC_COLLECTION_NAME, embedding=embeddings)

    search_filter = None
    if SHARED_DOCS_DIR not in selected_context:
        valid_files = collect_supported_files(selected_context)
        if valid_files:
            search_filter = rest.Filter(
                must=[rest.FieldCondition(key="metadata.source", match=rest.MatchAny(any=valid_files))]
            )

    retriever = vector_store.as_retriever(search_kwargs={"k": limit, "filter": search_filter})
    docs = retriever.invoke(query)

    results = []
    for doc in docs:
        metadata = doc.metadata or {}
        results.append(
            {
                "chunk_id": metadata.get("chunk_id", f"dense::{hash(doc.page_content)}"),
                "source": metadata.get("source", "Unknown"),
                "title": metadata.get("title", metadata.get("source", "Unknown")),
                "section_path": metadata.get("section_path", ""),
                "page": metadata.get("page"),
                "modality": metadata.get("modality", "text"),
                "summary": metadata.get("summary", ""),
                "text": doc.page_content,
                "raw_text": doc.page_content,
                "score": 0.0,
                "retrieval_mode": "dense",
            }
        )
    return results


def retrieve_memory(query: str) -> List[Dict[str, Any]]:
    embeddings = get_embeddings()
    store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=CHAT_MEMORY_COLLECTION,
        embedding=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": MEMORY_TOP_K})
    docs = retriever.invoke(query)
    results = []
    for idx, doc in enumerate(docs):
        results.append(
            {
                "chunk_id": f"memory::{idx}::{hash(doc.page_content)}",
                "source": "Past Conversation Memory",
                "title": "Past Conversation Memory",
                "section_path": "",
                "page": None,
                "modality": "memory",
                "summary": "",
                "text": doc.page_content,
                "raw_text": doc.page_content,
                "score": 0.0,
                "retrieval_mode": "memory",
            }
        )
    return results


@lru_cache(maxsize=1)
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def init_neo4j_indexes():
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            for label in ONTOLOGY.get("node_labels", []):
                constraint_name = f"node_{label.lower()}_id_unique"
                try:
                    session.run(f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
                except Exception as e:
                    # Constraint might already exist
                    pass
    except Exception as e:
        logger.error(f"Failed to create Neo4j indexes: {e}", exc_info=True)


def retrieve_graph_context(question: str) -> List[Dict[str, Any]]:
    llm = get_chat_llm(temperature=0.0, as_json=True)
    prompt = f"""Extract key entities from the following question to query a knowledge graph.
Return JSON only:
{{
  "entities": ["Entity1", "Entity2"]
}}

Question: {question}
"""
    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        entities = [str(e).strip().lower() for e in data.get("entities", []) if str(e).strip()]
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse graph entities JSON: {e}")
        entities = [token.lower() for token in question.split() if len(token) > 3]
    except Exception as e:
        logger.error(f"Unexpected error in retrieve_graph_context: {e}", exc_info=True)
        entities = [token.lower() for token in question.split() if len(token) > 3]

    if not entities:
        return []

    matches = []
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            query = '''
            UNWIND $entities AS entity
            MATCH p = (n)-[*1..2]-(m)
            WHERE n.id = entity
            RETURN nodes(p) AS nodes, relationships(p) AS rels
            LIMIT $limit
            '''
            result = session.run(query, entities=entities, limit=GRAPH_TOP_K * 2)
            
            paths_text = []
            for record in result:
                nodes = record["nodes"]
                rels = record["rels"]
                if not nodes or not rels:
                    continue
                path_str = ""
                for i, rel in enumerate(rels):
                    start_node = nodes[i].get("id", "Unknown")
                    end_node = nodes[i+1].get("id", "Unknown")
                    rel_type = rel.type
                    path_str += f"{start_node} -[{rel_type}]-> {end_node}  "
                paths_text.append(path_str.strip())

        if paths_text:
            text = "Knowledge Graph Multi-hop Evidence\n" + "\n".join(paths_text)
            matches.append(
                {
                    "chunk_id": f"graph::{hash(text)}",
                    "source": "Neo4j Knowledge Graph",
                    "title": "Neo4j Knowledge Graph",
                    "section_path": "",
                    "page": None,
                    "modality": "graph",
                    "summary": f"Graph multi-hop evidence for entities: {', '.join(entities)}",
                    "text": text,
                    "raw_text": text,
                    "score": 0.0,
                    "retrieval_mode": "graph",
                }
            )
    except Exception as e:
        logger.error(f"Neo4j Retrieval Error: {e}", exc_info=True)

    return matches[:GRAPH_TOP_K]


def reciprocal_rank_fusion(result_groups: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    for group in result_groups:
        for rank, result in enumerate(group, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["retrieval_modes"] = []
            fused[chunk_id]["rrf_score"] += rrf_score
            mode = result.get("retrieval_mode")
            if mode and mode not in fused[chunk_id]["retrieval_modes"]:
                fused[chunk_id]["retrieval_modes"].append(mode)
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def retrieve_hybrid(state: AgentState):
    plan = state.get("query_plan", {})
    search_queries = state.get("search_queries", [state["question"]])
    selected_context = state.get("selected_context", [SHARED_DOCS_DIR])
    selected_files = None if SHARED_DOCS_DIR in selected_context else collect_supported_files(selected_context)

    dense_groups = []
    lexical_groups = []
    visual_groups = []

    if "dense" in plan.get("retrieval_modes", []):
        for query in search_queries:
            dense_groups.append(dense_search(query, selected_context, DENSE_TOP_K))

    if "lexical" in plan.get("retrieval_modes", []):
        for query in search_queries:
            lexical_groups.append(lexical_search(query, selected_context, LEXICAL_TOP_K))

    if "visual" in plan.get("retrieval_modes", []):
        for query in search_queries:
            visual_groups.append(visual_search(query, selected_files, VISUAL_TOP_K))

    memory_group = retrieve_memory(state["question"]) if "memory" in plan.get("retrieval_modes", []) else []
    graph_group = retrieve_graph_context(state["question"]) if "graph" in plan.get("retrieval_modes", []) else []

    fused = reciprocal_rank_fusion(dense_groups + lexical_groups + visual_groups + [memory_group, graph_group], k=60)
    candidates = fused[:RERANK_TOP_K]

    final_docs = []
    if candidates:
        ranker = get_ranker()
        rerank_request = RerankRequest(
            query=state["question"],
            passages=[{"text": item["text"], "meta": item} for item in candidates],
        )
        reranked = ranker.rerank(rerank_request)[:FINAL_TOP_K]
        for item in reranked:
            meta = item["meta"]
            final_docs.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "source": meta["source"],
                    "title": meta.get("title", meta["source"]),
                    "section_path": meta.get("section_path", ""),
                    "page": meta.get("page"),
                    "modality": meta.get("modality", "text"),
                    "summary": meta.get("summary", ""),
                    "text": item["text"],
                    "raw_text": meta.get("raw_text", item["text"]),
                    "asset_path": meta.get("asset_path"),
                    "asset_type": meta.get("asset_type"),
                    "retrieval_modes": meta.get("retrieval_modes", [meta.get("retrieval_mode", "unknown")]),
                    "rrf_score": meta.get("rrf_score", 0.0),
                }
            )

    kg_context = "\n\n".join([doc["text"] for doc in graph_group[:GRAPH_TOP_K]])
    trace = {
        "query_plan": plan,
        "dense_candidate_count": sum(len(group) for group in dense_groups),
        "lexical_candidate_count": sum(len(group) for group in lexical_groups),
        "visual_candidate_count": sum(len(group) for group in visual_groups),
        "memory_candidate_count": len(memory_group),
        "graph_candidate_count": len(graph_group),
        "fused_candidate_count": len(fused),
        "final_sources": [doc["source"] for doc in final_docs],
        "visual_assets": [doc.get("asset_path") for doc in final_docs if doc.get("asset_path")],
    }
    return {"documents": final_docs, "kg_context": kg_context, "retrieval_trace": trace, "hallucination_count": 0}


def execute_tools(state: AgentState):
    tools_list = get_tools()
    tools_map = {t.name: t for t in tools_list}
    tool_calls = state.get("tool_calls", [])
    results = []
    
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        if name in tools_map:
            try:
                tool_output = tools_map[name].invoke(args)
                results.append({
                    "name": name,
                    "output": str(tool_output),
                    "args": args
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "output": f"Error executing tool: {e}",
                    "args": args
                })
    
    return {"tool_results": results}


def build_grounded_context(documents: List[Dict[str, Any]]) -> str:
    sections = []
    total_chars = 0
    for idx, doc in enumerate(documents, start=1):
        section = (
            f"[Evidence {idx}]\n"
            f"Source: {doc['source']}\n"
            f"Title: {doc.get('title', doc['source'])}\n"
            f"Section: {doc.get('section_path', '') or 'Unspecified'}\n"
            f"Page: {doc.get('page', 'N/A')}\n"
            f"Modality: {doc.get('modality', 'text')}\n"
            f"Retrieval Modes: {', '.join(doc.get('retrieval_modes', []))}\n"
            f"Content:\n{doc['text']}\n"
        )
        if total_chars + len(section) > MAX_CONTEXT_CHARS:
            break
        sections.append(section)
        total_chars += len(section)
    return "\n\n".join(sections)


async def generate_grounded(state: AgentState):
    llm = get_chat_llm(temperature=0.2, as_json=False).with_config(tags=["generate_grounded"])
    facts_text = fact_texts_for_prompt(state.get("user_profile", []))
    history = "\n".join(
        [f"{message['role']}: {message['content']}" for message in state.get("chat_history", [])[-MAX_CHAT_HISTORY_FOR_ANSWER:]]
    )
    grounded_context = build_grounded_context(state.get("documents", []))
    answer_style = state.get("query_plan", {}).get("answer_style", "normal")

    tool_results = state.get("tool_results", [])
    if tool_results:
        tool_results_text = "Live Tool Data (Priority Evidence):\n"
        for res in tool_results:
            tool_results_text += f"- Tool '{res['name']}': {res['output']}\n"
    else:
        tool_results_text = "None"

    prompt = f"""You are an intelligent Home Automation Manager for a local multimodal RAG system.
Your role is to contextually synthesize real-time sensor data, visual camera feeds, document context, and knowledge graph data to provide accurate and helpful answers.

Use the evidence below whenever evidence exists.
If live sensor data or tool output is provided in "Live Tool Data", trust it over historical documents.
If evidence is insufficient, say so explicitly.
Do NOT include a list of citations or sources in your answer text. Citations are handled by a separate UI element, so appending them here is redundant.

Before answering, you MUST wrap your step-by-step reasoning inside <think> and </think> tags.
IMPORTANT: You MUST close the reasoning with </think> and place your final user-facing answer OUTSIDE and AFTER the <think> tags.

Answer style: {answer_style}

User Profile Facts:
{facts_text}

Conversation History:
{history or 'None'}

Knowledge Graph Context:
{state.get('kg_context', '') or 'None'}

Live Tool Data:
{tool_results_text}

Evidence:
{grounded_context or 'No retrieved evidence'}

Question:
{state['question']}
"""
    try:
        response = await llm.ainvoke(prompt)
        answer_text = response.content
        answer_text = re.sub(r'(?i)(?:\n|^)#{0,3}\s*\**(?:Citations|Sources|References)\**:?.*', '', answer_text, flags=re.DOTALL).strip()
        clean_answer = re.sub(r'<think>.*?(?:</think>|$)', '', answer_text, flags=re.DOTALL).strip()
        payload = {
            "answer": answer_text,
            "citations": [],
            "confidence": 0.9,
            "insufficient_evidence": "insufficient evidence" in clean_answer.lower() or "not enough evidence" in clean_answer.lower(),
        }
    except Exception as e:
        logger.error(f"Unexpected error in generate_grounded: {e}", exc_info=True)
        answer_text = "I could not produce a grounded answer from the current evidence."
        payload = {
            "answer": answer_text,
            "citations": [],
            "confidence": 0.0,
            "insufficient_evidence": True,
        }

    return {
        "generation": answer_text,
        "answer_payload": payload,
    }


def update_retry(state: AgentState):
    current_plan = dict(state.get("query_plan", {}))
    search_queries = list(current_plan.get("search_queries", []))
    broadened = f"{state['question']} exact filenames config values errors tables figures screenshots"
    if broadened not in search_queries:
        search_queries.append(broadened)
    current_plan["search_queries"] = search_queries[:5]
    current_plan.setdefault("retrieval_modes", ["dense", "lexical", "memory", "graph", "visual"])
    return {
        "hallucination_count": state["hallucination_count"] + 1,
        "query_plan": current_plan,
        "search_queries": current_plan["search_queries"],
    }


def grade_grounding(state: AgentState):
    if state["hallucination_count"] >= MAX_HALLUCINATION_RETRIES:
        return "update_memory"

    answer_payload = state.get("answer_payload", {})
    if answer_payload.get("insufficient_evidence"):
        return "update_memory"

    llm = get_chat_llm(temperature=0.0, as_json=True)
    context = build_grounded_context(state.get("documents", []))
    
    tool_results = state.get("tool_results", [])
    if tool_results:
        tool_results_text = "Live Tool Data (Priority Evidence):\n"
        for res in tool_results:
            tool_results_text += f"- Tool '{res['name']}': {res['output']}\n"
    else:
        tool_results_text = "None"

    generation = state.get('generation', '')
    clean_generation = re.sub(r'<think>.*?(?:</think>|$)', '', generation, flags=re.DOTALL).strip()
        
    prompt = f"""Check whether the answer is grounded in the Evidence OR Live Tool Data.

Return JSON only:
{{"grounded": true}}

Rules:
- grounded = true if the answer is supported by the Combined Evidence (Docs & Live Tools), or explicitly states that evidence is insufficient.
- grounded = false if the answer adds unsupported factual claims.

Combined Evidence (Docs & Live Tools):
Live Tool Data:
{tool_results_text}

Document Evidence:
{context or 'No evidence'}

Answer:
{clean_generation}
"""
    try:
        response = llm.invoke(prompt)
        grounded = bool(json.loads(response.content).get("grounded", True))
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse grounding grade JSON: {e}")
        grounded = True
    except Exception as e:
        logger.error(f"Unexpected error in grade_grounding: {e}", exc_info=True)
        grounded = True
    return "update_memory" if grounded else "rewrite"


def update_memory(state: AgentState):
    embeddings = get_embeddings()
    store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=CHAT_MEMORY_COLLECTION,
        embedding=embeddings,
    )
    
    generation = state.get('generation', '')
    clean_generation = re.sub(r'<think>.*?(?:</think>|$)', '', generation, flags=re.DOTALL).strip()

    interaction = (
        f"Question: {state['question']}\n"
        f"Answer: {clean_generation}\n"
        f"Citations: {', '.join(state.get('answer_payload', {}).get('citations', [])) or 'None'}"
    )
    store.add_texts([interaction], metadatas=[{"source": "Semantic Memory"}])

    llm = get_chat_llm(temperature=0.0, as_json=True)
    ontology_str = json.dumps(ONTOLOGY, indent=2)
    prompt = f"""Extract durable user facts and strict knowledge graph triples from this interaction.

You MUST adhere to the following Ontology for all extracted triples:
{ontology_str}

A valid triple must be formatted as: ["NodeName", "NodeLabel", "RELATIONSHIP_TYPE", "TargetNodeName", "TargetNodeLabel"]
where NodeLabel and TargetNodeLabel are from the `node_labels` list, and RELATIONSHIP_TYPE is from the `relationship_types` list.

Return JSON only:
{{
  "facts": [{{"text": "User is building X", "confidence": 0.82}}],
  "triples": [["Sensor_A", "Sensor", "LOCATED_IN", "Fish_Tank_1", "Location"]]
}}

Only keep facts likely to matter in future conversations.
If there are no durable facts, return empty arrays.

Interaction:
Question: {state['question']}
Answer: {clean_generation}
"""
    existing_facts = normalize_fact_records(state.get("user_profile", []))
    new_facts = list(existing_facts)

    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        for fact in data.get("facts", []):
            if isinstance(fact, dict):
                text = str(fact.get("text", "")).strip()
                confidence = float(fact.get("confidence", 0.7))
            else:
                text = str(fact).strip()
                confidence = 0.7
            if text and all(existing["text"].lower() != text.lower() for existing in new_facts):
                new_facts.append({"text": text, "confidence": confidence})

        triples = data.get("triples", [])
        if triples:
            try:
                driver = get_neo4j_driver()
                with driver.session() as session:
                    for triple in triples:
                        if isinstance(triple, list) and len(triple) == 5:
                            subj, subj_label, rel, obj, obj_label = [str(item).strip() for item in triple]
                            if subj_label in ONTOLOGY.get("node_labels", []) and obj_label in ONTOLOGY.get("node_labels", []) and rel in ONTOLOGY.get("relationship_types", []):
                                if subj and obj:
                                    subj = subj.lower()
                                    obj = obj.lower()
                                    query = f'''
                                    MERGE (s:{subj_label} {{id: $subj}})
                                    MERGE (o:{obj_label} {{id: $obj}})
                                    MERGE (s)-[r:{rel}]->(o)
                                    '''
                                    session.run(query, subj=subj, obj=obj)
            except Exception as e:
                logger.error(f"Neo4j Error: {e}", exc_info=True)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse memory facts JSON: {e}")
    except Exception as e:
        logger.error(f"Error extracting memory facts or graph triples: {e}", exc_info=True)

    return {"user_profile": new_facts}


def build_app_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("plan_query", plan_query)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("retrieve_hybrid", retrieve_hybrid)
    workflow.add_node("generate_grounded", generate_grounded)
    workflow.add_node("update_retry", update_retry)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("plan_query")
    workflow.add_edge("plan_query", "retrieve_hybrid")
    workflow.add_edge("plan_query", "execute_tools")
    workflow.add_edge("retrieve_hybrid", "generate_grounded")
    workflow.add_edge("execute_tools", "generate_grounded")
    workflow.add_conditional_edges(
        "generate_grounded",
        grade_grounding,
        {"update_memory": "update_memory", "rewrite": "update_retry"},
    )
    workflow.add_edge("update_retry", "retrieve_hybrid")
    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=checkpointer)
