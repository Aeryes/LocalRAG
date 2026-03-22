import os
import json
import re

APP_TITLE = "LocalRAG Frontier"
APP_LAYOUT = "wide"
APP_CAPTION = "Hybrid Dense+Lexical Retrieval | Visual Asset Retrieval | Semantic Memory | GraphRAG | Multimodal-Ready"
DEFAULT_THREAD_ID = "thread_1"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
PHOENIX_PORT = os.getenv("PHOENIX_PORT", "6006")
PHOENIX_UI_URL = os.getenv("PHOENIX_UI_URL", f"http://localhost:{PHOENIX_PORT}")
FILE_MANAGER_URL = os.getenv("FILE_MANAGER_URL", "http://localhost:8080")

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_CHAT_MODEL)

VISUAL_EMBED_MODEL_NAME = os.getenv("VISUAL_EMBED_MODEL_NAME", "google/siglip2-base-patch16-224")
TABLE_DETECTION_MODEL_NAME = os.getenv("TABLE_DETECTION_MODEL_NAME", "microsoft/table-transformer-detection")

DOC_COLLECTION_NAME = "frontier_docs"
VISUAL_COLLECTION_NAME = "frontier_visual_assets"
CHAT_MEMORY_COLLECTION = "frontier_chat_memory"

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:password@localhost:5432/rag_db")

SHARED_DOCS_DIR = "/app/docs"
ASSETS_DIR = os.path.join(SHARED_DOCS_DIR, ".rag_assets")

ONTOLOGY_PATH = os.getenv("ONTOLOGY_PATH", "ontology.json")
try:
    with open(ONTOLOGY_PATH, "r") as f:
        ONTOLOGY = json.load(f)
except Exception:
    ONTOLOGY = {
        "node_labels": ["Sensor", "Location", "Entity", "Metric", "User", "Task"],
        "relationship_types": ["LOCATED_IN", "MONITORS", "HAS_STATE", "AFFECTS", "INTERACTS_WITH"]
    }

_cypher_safe_pattern = re.compile(r"^[A-Za-z0-9_]+$")
for label in ONTOLOGY.get("node_labels", []):
    if not _cypher_safe_pattern.match(label):
        raise ValueError(f"Invalid node label in ontology: {label}")
for rel in ONTOLOGY.get("relationship_types", []):
    if not _cypher_safe_pattern.match(rel):
        raise ValueError(f"Invalid relationship type in ontology: {rel}")

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "4096"))
VISUAL_VECTOR_SIZE = int(os.getenv("VISUAL_VECTOR_SIZE", "768"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
CONTEXT_WINDOW_CHARS = int(os.getenv("CONTEXT_WINDOW_CHARS", "220"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
LEXICAL_TOP_K = int(os.getenv("LEXICAL_TOP_K", "10"))
VISUAL_TOP_K = int(os.getenv("VISUAL_TOP_K", "6"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "4"))
GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "12"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "6"))

TABLE_DETECTION_THRESHOLD = float(os.getenv("TABLE_DETECTION_THRESHOLD", "0.92"))
MAX_TABLES_PER_PAGE = int(os.getenv("MAX_TABLES_PER_PAGE", "5"))
PDF_RENDER_SCALE = float(os.getenv("PDF_RENDER_SCALE", "2.0"))

MAX_QUERY_HISTORY = 4
MAX_CHAT_HISTORY_FOR_ANSWER = 6
MAX_HALLUCINATION_RETRIES = 2
STREAM_DELAY_SECONDS = 0.01

SUPPORTED_TEXT_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")
SUPPORTED_VISUAL_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_ALL_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_VISUAL_EXTENSIONS

BLOCKED_EXACT_FILENAMES = {
    "app.py",
    "constants.py",
    "services.py",
    "graph.py",
    "ui.py",
    "multimodal.py",
    "docker-compose.yml",
    "requirements.txt",
    "makefile",
    "dockerfile",
}
BLOCKED_FILE_SUFFIXES = (".py", ".yml", ".graphml", ".sqlite")

USER_AVATAR = "🟢"
ASSISTANT_AVATAR = "🟡"

SIDEBAR_CONTEXT_HEADER = "🎯 Context Filter"
SIDEBAR_MEMORY_HEADER = "🧠 Agent Memory State"
SIDEBAR_DEV_HEADER = "🛠️ Developer Tools"

DEBUG_TOGGLE_LABEL = "Debug mode"
DEBUG_TOGGLE_HELP = "Show developer tools, observability links, and multimodal retrieval trace details."
OBSERVABILITY_BUTTON_LABEL = "📈 Open Observability (Arize Phoenix)"
FILE_MANAGER_BUTTON_LABEL = "📂 Open File Manager"
RESET_BUTTON_LABEL = "🗑️ Reset Retrieval State"

EMPTY_FACTS_TEXT = "No extracted facts yet."