import os
import logging
import threading
import uuid
from typing import Dict, List, Sequence
from functools import lru_cache

import psycopg_pool
from psycopg.rows import dict_row

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from constants import (
    ASSETS_DIR,
    BLOCKED_EXACT_FILENAMES,
    BLOCKED_FILE_SUFFIXES,
    CHAT_MEMORY_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOC_COLLECTION_NAME,
    POSTGRES_DSN,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    QDRANT_URL,
    SHARED_DOCS_DIR,
    SUPPORTED_ALL_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    VECTOR_SIZE,
    VISUAL_COLLECTION_NAME,
    VISUAL_VECTOR_SIZE,
)
from multimodal import build_visual_assets_for_file, delete_asset_directory_for_source, upsert_visual_assets

logger = logging.getLogger(__name__)


db_pool = psycopg_pool.ConnectionPool(
    conninfo=POSTGRES_DSN,
    min_size=1,
    max_size=10,
    kwargs={"row_factory": dict_row}
)

@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def init_postgres_indexes() -> None:
    with db_pool.connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_registry (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT,
                section_path TEXT,
                page INTEGER,
                modality TEXT NOT NULL,
                summary TEXT,
                contextual_text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(section_path, '') || ' ' || coalesce(summary, '') || ' ' || coalesce(contextual_text, '') || ' ' || coalesce(raw_text, ''))) STORED
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS chunk_registry_fts_idx ON chunk_registry USING GIN (fts)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_asset_registry (
                asset_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                asset_path TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                page INTEGER,
                bbox TEXT,
                description TEXT,
                text_context TEXT
            )
            """
        )
        connection.commit()


def init_qdrant_collections() -> None:
    client = get_qdrant_client()

    if not client.collection_exists(DOC_COLLECTION_NAME):
        client.create_collection(
            collection_name=DOC_COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
        )

    if not client.collection_exists(VISUAL_COLLECTION_NAME):
        client.create_collection(
            collection_name=VISUAL_COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=VISUAL_VECTOR_SIZE, distance=rest.Distance.COSINE),
        )

    if not client.collection_exists(CHAT_MEMORY_COLLECTION):
        client.create_collection(
            collection_name=CHAT_MEMORY_COLLECTION,
            vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
        )


def init_storage() -> None:
    os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    init_qdrant_collections()
    init_postgres_indexes()


def load_local_file(file_path: str):
    lowered_path = file_path.lower()
    try:
        if lowered_path.endswith(".pdf"):
            return PyMuPDFLoader(file_path).load()
        if lowered_path.endswith(".docx"):
            return Docx2txtLoader(file_path).load()
        return TextLoader(file_path, encoding="utf-8").load()
    except Exception as exc:
        logger.error(f"File Load Error for {file_path}: {exc}", exc_info=True)
        return []


def is_supported_path(file_path: str) -> bool:
    return file_path.lower().endswith(SUPPORTED_ALL_EXTENSIONS)


def is_text_indexable(file_path: str) -> bool:
    return file_path.lower().endswith(SUPPORTED_TEXT_EXTENSIONS)


def is_safe_path(path: str, base_dir: str = SHARED_DOCS_DIR) -> bool:
    try:
        base_abs = os.path.abspath(os.path.realpath(base_dir)).replace('\\', '/')
        path_abs = os.path.abspath(os.path.realpath(path)).replace('\\', '/')
        return path_abs.startswith(base_abs)
    except Exception:
        return False


def collect_supported_files(selected_paths: Sequence[str]) -> List[str]:
    files_to_check: List[str] = []
    for path in selected_paths:
        if not is_safe_path(path):
            logger.warning(f"Security Warning: Path traversal attempt blocked for path: {path}")
            continue
        if os.path.isfile(path):
            if is_supported_path(path):
                files_to_check.append(os.path.abspath(os.path.realpath(path)).replace('\\', '/'))
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    if is_supported_path(full_path):
                        files_to_check.append(os.path.abspath(os.path.realpath(full_path)).replace('\\', '/'))
    return list(sorted(set(files_to_check)))


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def derive_section_path(source: str, text: str) -> str:
    if source.lower().endswith(".md"):
        headings = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                headings.append(stripped.lstrip("#").strip())
                if len(headings) >= 2:
                    break
        if headings:
            return " > ".join(headings)
    return ""


def estimate_chunk_summary(text: str) -> str:
    cleaned = normalize_whitespace(text)
    return cleaned[:220]


def contextualize_chunk(*, source: str, title: str, section_path: str, page: int, summary: str, raw_text: str) -> str:
    prefix_lines = [
        f"Document: {title}",
        f"Source: {source}",
        f"Page: {page if page is not None else 'N/A'}",
        f"Section: {section_path or 'Unspecified'}",
        f"Local Summary: {summary or 'None'}",
    ]
    return "\n".join(prefix_lines) + f"\n\n{raw_text}"


def build_structured_text_chunks(file_path: str) -> List[Dict]:
    if not is_text_indexable(file_path):
        return []

    docs = load_local_file(file_path)
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    title = os.path.basename(file_path)
    structured_chunks: List[Dict] = []

    for idx, doc in enumerate(split_docs):
        raw_text = normalize_whitespace(doc.page_content)
        if not raw_text:
            continue

        metadata = doc.metadata or {}
        page = metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else None
        section_path = derive_section_path(file_path, raw_text[:300])
        summary = estimate_chunk_summary(raw_text[:400])
        contextual_text = contextualize_chunk(
            source=file_path,
            title=title,
            section_path=section_path,
            page=page_num,
            summary=summary,
            raw_text=raw_text,
        )
        content_hash = str(abs(hash(contextual_text)))
        chunk_id = f"{file_path}::chunk::{idx}::{content_hash}"

        structured_chunks.append(
            {
                "chunk_id": chunk_id,
                "source": file_path,
                "title": title,
                "section_path": section_path,
                "page": page_num,
                "modality": "text",
                "summary": summary,
                "contextual_text": contextual_text,
                "raw_text": raw_text,
                "content_hash": content_hash,
            }
        )

    return structured_chunks


def delete_file_context(file_path: str) -> None:
    if not is_safe_path(file_path):
        logger.warning(f"Security Warning: Path traversal attempt blocked for path: {file_path}")
        return

    file_path = os.path.abspath(os.path.realpath(file_path)).replace('\\', '/')
    client = get_qdrant_client()

    if client.collection_exists(DOC_COLLECTION_NAME):
        try:
            client.delete(
                collection_name=DOC_COLLECTION_NAME,
                points_selector=rest.Filter(
                    must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=file_path))]
                ),
            )
        except Exception:
            pass

    if client.collection_exists(VISUAL_COLLECTION_NAME):
        try:
            client.delete(
                collection_name=VISUAL_COLLECTION_NAME,
                points_selector=rest.Filter(
                    must=[rest.FieldCondition(key="source", match=rest.MatchValue(value=file_path))]
                ),
            )
        except Exception:
            pass

    with db_pool.connection() as connection:
        connection.execute("DELETE FROM chunk_registry WHERE source = %s", (file_path,))
        connection.execute("DELETE FROM visual_asset_registry WHERE source = %s", (file_path,))
        connection.commit()

    delete_asset_directory_for_source(file_path)


def upsert_text_chunks(chunks: List[Dict]) -> None:
    if not chunks:
        return

    embeddings = get_embeddings()
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=DOC_COLLECTION_NAME,
        embedding=embeddings,
    )

    texts = [chunk["contextual_text"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "title": chunk["title"],
            "section_path": chunk["section_path"],
            "page": chunk["page"],
            "modality": chunk["modality"],
            "summary": chunk["summary"],
        }
        for chunk in chunks
    ]
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])) for chunk in chunks]
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    with db_pool.connection() as connection:
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO chunk_registry
                (chunk_id, source, title, section_path, page, modality, summary, contextual_text, raw_text, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    source = EXCLUDED.source,
                    title = EXCLUDED.title,
                    section_path = EXCLUDED.section_path,
                    page = EXCLUDED.page,
                    modality = EXCLUDED.modality,
                    summary = EXCLUDED.summary,
                    contextual_text = EXCLUDED.contextual_text,
                    raw_text = EXCLUDED.raw_text,
                    content_hash = EXCLUDED.content_hash
                """,
                (
                    chunk["chunk_id"],
                    chunk["source"],
                    chunk["title"],
                    chunk["section_path"],
                    chunk["page"],
                    chunk["modality"],
                    chunk["summary"],
                    chunk["contextual_text"],
                    chunk["raw_text"],
                    chunk["content_hash"],
                ),
            )
        connection.commit()


def upsert_visual_assets_to_registry(assets: List[Dict]) -> None:
    if not assets:
        return

    with db_pool.connection() as connection:
        for asset in assets:
            connection.execute(
                """
                INSERT INTO visual_asset_registry
                (asset_id, source, asset_path, asset_type, page, bbox, description, text_context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                    source = EXCLUDED.source,
                    asset_path = EXCLUDED.asset_path,
                    asset_type = EXCLUDED.asset_type,
                    page = EXCLUDED.page,
                    bbox = EXCLUDED.bbox,
                    description = EXCLUDED.description,
                    text_context = EXCLUDED.text_context
                """,
                (
                    asset["asset_id"],
                    asset["source"],
                    asset["asset_path"],
                    asset["asset_type"],
                    asset["page"],
                    asset["bbox"],
                    asset["description"],
                    asset["text_context"],
                ),
            )
        connection.commit()


def index_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        return

    file_path = os.path.abspath(os.path.realpath(file_path)).replace('\\', '/')
    delete_file_context(file_path)

    text_chunks = build_structured_text_chunks(file_path)
    if text_chunks:
        upsert_text_chunks(text_chunks)

    visual_assets = build_visual_assets_for_file(file_path)
    if visual_assets:
        upsert_visual_assets(visual_assets)
        upsert_visual_assets_to_registry(visual_assets)


class DocumentSyncHandler(FileSystemEventHandler):
    def process_file(self, file_path: str) -> None:
        file_path = os.path.abspath(os.path.realpath(file_path)).replace('\\', '/')
        filename = os.path.basename(file_path).lower()
        if filename in BLOCKED_EXACT_FILENAMES:
            return
        if filename.endswith(BLOCKED_FILE_SUFFIXES):
            return
        if not filename.endswith(SUPPORTED_ALL_EXTENSIONS):
            return
        try:
            index_file(file_path)
        except Exception as exc:
            logger.error(f"Sync Error (Index): {exc}", exc_info=True)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            delete_file_context(os.path.abspath(os.path.realpath(event.src_path)).replace('\\', '/'))


def start_watchdog():
    os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    observer = PollingObserver()
    observer.schedule(DocumentSyncHandler(), SHARED_DOCS_DIR, recursive=True)
    threading.Thread(target=observer.start, daemon=True).start()
    return observer


def ensure_selected_context_indexed(selected_paths: Sequence[str]) -> None:
    for file_path in collect_supported_files(selected_paths):
        with db_pool.connection() as connection:
            has_text = connection.execute(
                "SELECT chunk_id FROM chunk_registry WHERE source = %s LIMIT 1",
                (file_path,),
            ).fetchone()
            has_visual = connection.execute(
                "SELECT asset_id FROM visual_asset_registry WHERE source = %s LIMIT 1",
                (file_path,),
            ).fetchone()

        if has_text is None and has_visual is None:
            index_file(file_path)


def lexical_search(query: str, selected_context: Sequence[str], limit: int) -> List[Dict]:
    with db_pool.connection() as connection:
        context_paths = tuple(collect_supported_files(selected_context)) if SHARED_DOCS_DIR not in selected_context else None
        
        if context_paths is not None:
            if not context_paths:
                return []
            rows = connection.execute(
                """
                SELECT
                    chunk_id, source, title, section_path, page, modality, summary, contextual_text, raw_text,
                    ts_rank(fts, plainto_tsquery('english', %s)) AS score
                FROM chunk_registry
                WHERE fts @@ plainto_tsquery('english', %s)
                  AND source = ANY(%s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, list(context_paths), limit * 3)
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT
                    chunk_id, source, title, section_path, page, modality, summary, contextual_text, raw_text,
                    ts_rank(fts, plainto_tsquery('english', %s)) AS score
                FROM chunk_registry
                WHERE fts @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, limit * 3)
            ).fetchall()

    results = []
    for row in rows:
        if context_paths is not None and row["source"] not in context_paths:
            continue
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "title": row["title"],
                "section_path": row["section_path"],
                "page": row["page"],
                "modality": row["modality"],
                "summary": row["summary"],
                "text": row["contextual_text"],
                "raw_text": row["raw_text"],
                "score": float(row["score"]) if row["score"] is not None else 0.0,
                "retrieval_mode": "lexical",
            }
        )
        if len(results) >= limit:
            break
    return results


def get_lexical_chunk_count() -> int:
    with db_pool.connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM chunk_registry").fetchone()
        return int(row["count"]) if row else 0


def get_visual_asset_count() -> int:
    with db_pool.connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM visual_asset_registry").fetchone()
        return int(row["count"]) if row else 0


def reset_all_data() -> None:
    client = get_qdrant_client()

    for collection_name in [DOC_COLLECTION_NAME, VISUAL_COLLECTION_NAME, CHAT_MEMORY_COLLECTION]:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

    try:
        with db_pool.connection() as connection:
            connection.execute("TRUNCATE TABLE chunk_registry, visual_asset_registry")
            connection.commit()
    except Exception:
        pass

    if os.path.exists(ASSETS_DIR):
        for root, dirs, files in os.walk(ASSETS_DIR, topdown=False):
            for file_name in files:
                os.remove(os.path.join(root, file_name))
            for dir_name in dirs:
                os.rmdir(os.path.join(root, dir_name))
        try:
            os.rmdir(ASSETS_DIR)
        except OSError:
            pass

    init_storage()
