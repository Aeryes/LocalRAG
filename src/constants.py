import os

APP_TITLE = "LocalRAG"
APP_LAYOUT = "wide"
APP_CAPTION = "Sqlite Checkpointing | Semantic Recall | Fact Extraction | GraphRAG"
DEFAULT_THREAD_ID = "thread_1"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
PHOENIX_PORT = os.getenv("PHOENIX_PORT", "6006")
PHOENIX_UI_URL = os.getenv("PHOENIX_UI_URL", f"http://localhost:{PHOENIX_PORT}")
FILE_MANAGER_URL = os.getenv("FILE_MANAGER_URL", "http://localhost:8080")

OLLAMA_MODEL = "llama3:8b"
COLLECTION_NAME = "general_docs"
CHAT_MEMORY_COLLECTION = "chat_history"
CHECKPOINT_DB = "checkpoints.sqlite"
SHARED_DOCS_DIR = "/app/docs"
KG_FILENAME = "knowledge_graph.graphml"
KG_PATH = os.path.join(SHARED_DOCS_DIR, KG_FILENAME)

VECTOR_SIZE = 4096
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DOC_RETRIEVAL_K = 5
CHAT_MEMORY_K = 2
RERANK_TOP_K = 4
MAX_KG_EDGES = 5
MAX_QUERY_HISTORY = 3
MAX_CHAT_HISTORY_FOR_ANSWER = 5
MAX_HALLUCINATION_RETRIES = 3
STREAM_DELAY_SECONDS = 0.015

SUPPORTED_DOC_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")
BLOCKED_EXACT_FILENAMES = {
    "app.py",
    "constants.py",
    "docker-compose.yml",
    "requirements.txt",
    "makefile",
    "dockerfile",
}
BLOCKED_FILE_SUFFIXES = (".py", ".yml", ".graphml")

USER_AVATAR = "🟢"
ASSISTANT_AVATAR = "🟡"

SIDEBAR_CONTEXT_HEADER = "🎯 Context Filter"
SIDEBAR_MEMORY_HEADER = "🧠 Agent Memory State"
SIDEBAR_DEV_HEADER = "🛠️ Developer Tools"
DEBUG_TOGGLE_LABEL = "Debug mode"
DEBUG_TOGGLE_HELP = "Show developer tools, observability links, and trace details."
OBSERVABILITY_BUTTON_LABEL = "📈 Open Observability (Arize Phoenix)"
FILE_MANAGER_BUTTON_LABEL = "📂 Open File Manager"
RESET_BUTTON_LABEL = "🗑️ Reset All Databases"

EMPTY_FACTS_TEXT = "No extracted facts yet."
DEV_TOOLS_HIDDEN_TEXT = "Developer tools are hidden."
VECTOR_DB_SUCCESS_TEMPLATE = "📊 Vector DB contains **{points_count}** chunks."
VECTOR_DB_EMPTY_TEXT = "📊 Vector DB is currently empty."

CHAT_INPUT_PLACEHOLDER = "Ask a question about your documents..."
ASSISTANT_SPINNER_TEXT = "Agent routing, retrieving, verifying, and updating memory..."
DEBUG_TRACE_LABEL = "Debug Trace"
EXTRACTED_FACTS_LABEL = "**Extracted Facts:**"