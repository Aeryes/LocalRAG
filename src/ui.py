import os
import threading
from typing import Iterable, List, Sequence

import networkx as nx
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from constants import (
    BLOCKED_EXACT_FILENAMES,
    BLOCKED_FILE_SUFFIXES,
    CHAT_MEMORY_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    KG_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    QDRANT_URL,
    SHARED_DOCS_DIR,
    SUPPORTED_DOC_EXTENSIONS,
    VECTOR_SIZE,
)


def init_qdrant_collections() -> None:
    client = QdrantClient(url=QDRANT_URL)
    for collection_name in [COLLECTION_NAME, CHAT_MEMORY_COLLECTION]:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
            )


def init_knowledge_graph() -> None:
    if not os.path.exists(KG_PATH):
        os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
        graph = nx.DiGraph()
        nx.write_graphml(graph, KG_PATH)


def load_local_file(file_path: str):
    lowered_path = file_path.lower()
    try:
        if lowered_path.endswith(".pdf"):
            return PyMuPDFLoader(file_path).load()
        if lowered_path.endswith(".docx"):
            return Docx2txtLoader(file_path).load()
        return TextLoader(file_path, encoding="utf-8").load()
    except Exception as exc:
        print(f"File Load Error for {file_path}: {exc}")
        return []


def is_supported_document(file_path: str) -> bool:
    return file_path.lower().endswith(SUPPORTED_DOC_EXTENSIONS)


def collect_supported_files(selected_paths: Sequence[str]) -> List[str]:
    files_to_check = []
    for path in selected_paths:
        if os.path.isfile(path):
            if is_supported_document(path):
                files_to_check.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    if is_supported_document(full_path):
                        files_to_check.append(full_path)
    return list(set(files_to_check))


class DocumentSyncHandler(FileSystemEventHandler):
    def delete_file_context(self, file_path: str) -> None:
        try:
            client = QdrantClient(url=QDRANT_URL)
            if client.collection_exists(COLLECTION_NAME):
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="metadata.source",
                                match=rest.MatchValue(value=file_path),
                            )
                        ]
                    ),
                )
        except Exception:
            pass

    def process_file(self, file_path: str) -> None:
        filename = os.path.basename(file_path).lower()
        if filename in BLOCKED_EXACT_FILENAMES:
            return
        if filename.endswith(BLOCKED_FILE_SUFFIXES):
            return
        if not filename.endswith(SUPPORTED_DOC_EXTENSIONS):
            return

        self.delete_file_context(file_path)

        if os.path.exists(file_path):
            try:
                docs = load_local_file(file_path)
                if not docs:
                    return

                splits = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                ).split_documents(docs)

                if not splits:
                    return

                embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
                QdrantVectorStore.from_documents(
                    splits,
                    embeddings,
                    url=QDRANT_URL,
                    collection_name=COLLECTION_NAME,
                )
            except Exception as exc:
                print(f"Sync Error (Index): {exc}")

    def on_created(self, event) -> None:
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self.delete_file_context(event.src_path)


def start_watchdog():
    os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
    observer = PollingObserver()
    observer.schedule(DocumentSyncHandler(), SHARED_DOCS_DIR, recursive=True)
    threading.Thread(target=observer.start, daemon=True).start()
    return observer


def ensure_selected_context_indexed(selected_paths: Sequence[str]) -> None:
    client = QdrantClient(url=QDRANT_URL)
    handler = DocumentSyncHandler()

    for file_path in collect_supported_files(selected_paths):
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=rest.Filter(
                must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=file_path))]
            ),
            limit=1,
        )
        if not records:
            handler.process_file(file_path)


def reset_all_data() -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if client.collection_exists(CHAT_MEMORY_COLLECTION):
        client.delete_collection(CHAT_MEMORY_COLLECTION)

    if os.path.exists(KG_PATH):
        os.remove(KG_PATH)

    os.makedirs(SHARED_DOCS_DIR, exist_ok=True)
    nx.write_graphml(nx.DiGraph(), KG_PATH)
    init_qdrant_collections()