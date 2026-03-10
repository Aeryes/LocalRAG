import sqlite3
import time

import phoenix as px
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from qdrant_client import QdrantClient

from constants import (
    APP_CAPTION,
    APP_LAYOUT,
    APP_TITLE,
    ASSISTANT_SPINNER_TEXT,
    AUTO_PUSH_SPINNER_TEXT,
    CHAT_INPUT_PLACEHOLDER,
    CHECKPOINT_DB,
    COLLECTION_NAME,
    DEBUG_TOGGLE_HELP,
    DEBUG_TOGGLE_LABEL,
    DEV_TOOLS_HIDDEN_TEXT,
    FILE_MANAGER_BUTTON_LABEL,
    FILE_MANAGER_URL,
    OBSERVABILITY_BUTTON_LABEL,
    PHOENIX_UI_URL,
    RESET_BUTTON_LABEL,
    SHARED_DOCS_DIR,
    SIDEBAR_CONTEXT_HEADER,
    SIDEBAR_DEV_HEADER,
    SIDEBAR_MEMORY_HEADER,
    STREAM_DELAY_SECONDS,
    VECTOR_DB_EMPTY_TEXT,
    VECTOR_DB_SUCCESS_TEMPLATE,
)
from graph import build_app_graph
from services import (
    ensure_selected_context_indexed,
    init_knowledge_graph,
    init_qdrant_collections,
    reset_all_data,
    start_watchdog,
)
from ui import (
    ensure_session_state_defaults,
    get_paths,
    get_checkpoint_values,
    render_debug_trace,
    render_facts,
    role_avatar,
    sync_profile_cache_from_checkpoint,
)

st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)


# --- Initial Setup & Observability ---
if "phoenix_session" not in st.session_state:
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    st.session_state["phoenix_session"] = px.launch_app()


# --- Session State ---
ensure_session_state_defaults()


# --- Service Initialization ---
init_qdrant_collections()
init_knowledge_graph()


@st.cache_resource
def get_watchdog_observer():
    return start_watchdog()


get_watchdog_observer()


# --- Graph Construction (With Checkpointing) ---
connection = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
memory = SqliteSaver(connection)
app_graph = build_app_graph(memory)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

checkpoint_values = sync_profile_cache_from_checkpoint(app_graph, config)


# --- Main UI ---
st.title(APP_TITLE)
st.caption(APP_CAPTION)

with st.sidebar:
    st.header(SIDEBAR_CONTEXT_HEADER)
    all_paths = get_paths(SHARED_DOCS_DIR)

    selected = st.multiselect(
        "Focus search on (Select multiple):",
        options=all_paths,
        default=st.session_state.selected_context if st.session_state.selected_context != [SHARED_DOCS_DIR] else [],
        format_func=lambda path: path.replace(SHARED_DOCS_DIR, "ROOT"),
        help="Leave blank to search everything, or pick specific folders/files.",
    )

    st.session_state.selected_context = selected if selected else [SHARED_DOCS_DIR]

    if selected:
        with st.spinner(AUTO_PUSH_SPINNER_TEXT):
            ensure_selected_context_indexed(selected)

    st.markdown("---")
    st.header(SIDEBAR_MEMORY_HEADER)
    facts_placeholder = st.empty()
    render_facts(
        facts_placeholder,
        st.session_state.user_profile_cache or checkpoint_values.get("user_profile", []),
    )

    st.markdown("---")
    st.toggle(
        DEBUG_TOGGLE_LABEL,
        key="debug_mode",
        help=DEBUG_TOGGLE_HELP,
    )

    if st.session_state.debug_mode:
        st.header(SIDEBAR_DEV_HEADER)
        st.link_button(OBSERVABILITY_BUTTON_LABEL, PHOENIX_UI_URL, use_container_width=True)
        st.link_button(FILE_MANAGER_BUTTON_LABEL, FILE_MANAGER_URL, use_container_width=True)

        try:
            from constants import QDRANT_URL
            collection_info = QdrantClient(url=QDRANT_URL).get_collection(COLLECTION_NAME)
            st.success(VECTOR_DB_SUCCESS_TEMPLATE.format(points_count=collection_info.points_count))
        except Exception:
            st.warning(VECTOR_DB_EMPTY_TEXT)

        if st.button(RESET_BUTTON_LABEL, use_container_width=True):
            reset_all_data()
            st.session_state.messages = []
            st.session_state.user_profile_cache = []
            st.session_state.last_trace = {}
            st.rerun()
    else:
        st.caption(DEV_TOOLS_HIDDEN_TEXT)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=role_avatar(message["role"])):
        st.markdown(message["content"])

if prompt := st.chat_input(CHAT_INPUT_PLACEHOLDER):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=role_avatar("user")):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=role_avatar("assistant")):
        with st.spinner(ASSISTANT_SPINNER_TEXT):
            final_state = app_graph.invoke(
                {
                    "question": prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "search_queries": [],
                    "documents": [],
                    "generation": "",
                    "hallucination_count": 0,
                    "user_profile": st.session_state.user_profile_cache,
                    "kg_context": "",
                    "selected_context": st.session_state.selected_context,
                },
                config=config,
            )

        latest_values = get_checkpoint_values(app_graph, config)
        latest_facts = final_state.get("user_profile") or latest_values.get("user_profile", [])
        st.session_state.user_profile_cache = latest_facts
        render_facts(facts_placeholder, latest_facts)

        st.session_state.last_trace = {
            "kg_context": final_state.get("kg_context", "None"),
            "search_queries": final_state.get("search_queries", []),
            "document_count": len(final_state.get("documents", [])),
            "hallucination_count": final_state.get("hallucination_count", 0),
        }

        def stream_text():
            for word in final_state["generation"].split(" "):
                yield word + " "
                time.sleep(STREAM_DELAY_SECONDS)

        st.write_stream(stream_text)

        if st.session_state.debug_mode:
            render_debug_trace(st.session_state.last_trace)

    st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})