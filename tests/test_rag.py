import pytest


def test_app_module_imports_successfully(app_module):
    assert hasattr(app_module, "transform_query")
    assert hasattr(app_module, "retrieve")
    assert hasattr(app_module, "generate")
    assert hasattr(app_module, "update_memory")
    assert hasattr(app_module, "DocumentSyncHandler")


def test_transform_query_returns_expanded_queries_and_original_question(app_module):
    state = {
        "question": "What database are we using?",
        "chat_history": [{"role": "user", "content": "Tell me about storage"}],
        "search_queries": [],
        "documents": [],
        "generation": "",
        "hallucination_count": 0,
        "user_profile": [],
        "kg_context": "",
    }

    result = app_module.transform_query(state)

    assert "search_queries" in result
    assert isinstance(result["search_queries"], list)
    assert len(result["search_queries"]) >= 3
    assert state["question"] in result["search_queries"]


def test_retrieve_returns_ranked_documents(app_module):
    app_module.st.session_state["selected_context"] = [app_module.SHARED_DOCS_DIR]

    state = {
        "question": "What database are we using?",
        "chat_history": [],
        "search_queries": ["What database are we using?"],
        "documents": [],
        "generation": "",
        "hallucination_count": 0,
        "user_profile": [],
        "kg_context": "",
    }

    result = app_module.retrieve(state)

    assert "documents" in result
    assert isinstance(result["documents"], list)
    assert len(result["documents"]) >= 1
    assert any("Qdrant" in doc["text"] for doc in result["documents"])
    assert all("source" in doc for doc in result["documents"])


def test_generate_returns_string_answer(app_module):
    state = {
        "question": "What database are we using?",
        "chat_history": [{"role": "user", "content": "What database are we using?"}],
        "search_queries": [],
        "documents": [
            {
                "text": "The application uses Qdrant as a vector database in a Docker container.",
                "source": "architecture.md",
            }
        ],
        "generation": "",
        "hallucination_count": 0,
        "user_profile": [],
        "kg_context": "",
    }

    result = app_module.generate(state)

    assert "generation" in result
    assert isinstance(result["generation"], str)
    assert "Qdrant" in result["generation"]


def test_grade_hallucination_routes_to_update_memory_when_judge_passes(app_module):
    state = {
        "question": "What database are we using?",
        "chat_history": [],
        "search_queries": [],
        "documents": [{"text": "Qdrant is the vector database.", "source": "architecture.md"}],
        "generation": "Qdrant is the vector database.",
        "hallucination_count": 0,
        "user_profile": [],
        "kg_context": "",
    }

    route = app_module.grade_hallucination(state)

    assert route == "update_memory"


def test_grade_hallucination_stops_retry_loop_after_max_retries(app_module):
    state = {
        "question": "What database are we using?",
        "chat_history": [],
        "search_queries": [],
        "documents": [],
        "generation": "Some answer",
        "hallucination_count": 3,
        "user_profile": [],
        "kg_context": "",
    }

    route = app_module.grade_hallucination(state)

    assert route == "update_memory"


def test_update_memory_merges_facts_without_duplicates(app_module):
    state = {
        "question": "How are we testing this?",
        "chat_history": [],
        "search_queries": [],
        "documents": [],
        "generation": "We are testing CI behavior.",
        "hallucination_count": 0,
        "user_profile": ["User is testing CI"],
        "kg_context": "",
    }

    result = app_module.update_memory(state)

    assert "user_profile" in result
    assert result["user_profile"].count("User is testing CI") == 1


def test_update_retry_increments_counter(app_module):
    state = {
        "hallucination_count": 1,
    }

    result = app_module.update_retry(state)

    assert result["hallucination_count"] == 2


def test_document_sync_handler_ignores_blocked_project_files(app_module, monkeypatch):
    handler = app_module.DocumentSyncHandler()
    calls = {"delete_called": False}

    def fake_delete(_file_path):
        calls["delete_called"] = True

    monkeypatch.setattr(handler, "delete_file_context", fake_delete)

    handler.process_file("app.py")

    assert calls["delete_called"] is False


def test_document_sync_handler_processes_supported_text_files(app_module, fake_doc_class, monkeypatch, tmp_path):
    handler = app_module.DocumentSyncHandler()
    test_file = tmp_path / "notes.txt"
    test_file.write_text("This is a test document.", encoding="utf-8")

    captured = {"indexed": False}

    def fake_load_local_file(file_path):
        return [fake_doc_class("This is a test document.", {"source": file_path})]

    class FakeVectorStoreForAssertion:
        @classmethod
        def from_documents(cls, docs, embeddings, url=None, collection_name=None):
            captured["indexed"] = True
            captured["doc_count"] = len(docs)
            captured["collection_name"] = collection_name
            return cls()

    monkeypatch.setattr(app_module, "load_local_file", fake_load_local_file)
    monkeypatch.setattr(app_module, "QdrantVectorStore", FakeVectorStoreForAssertion)
    monkeypatch.setattr(handler, "delete_file_context", lambda _: None)

    handler.process_file(str(test_file))

    assert captured["indexed"] is True
    assert captured["doc_count"] == 1
    assert captured["collection_name"] == app_module.COLLECTION_NAME