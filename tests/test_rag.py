import pytest
from fastapi.testclient import TestClient

def test_app_module_imports_successfully(app_module):
    assert hasattr(app_module, "app")
    assert hasattr(app_module, "app_graph")


def test_get_context_requires_auth(app_module):
    client = TestClient(app_module.app)
    response = client.get("/api/context")
    assert response.status_code == 401


def test_get_context_success(app_module, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = TestClient(app_module.app)
    response = client.get("/api/context", headers={"X-API-Key": "test-key", "X-Session-ID": "test-session"})
    assert response.status_code == 200
    assert "all_paths" in response.json()
    assert "selected_context" in response.json()


def test_update_context_success(app_module, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = TestClient(app_module.app)
    from src.constants import SHARED_DOCS_DIR
    import os
    safe_path = os.path.join(SHARED_DOCS_DIR, "safe_subfolder").replace('\\', '/')
    response = client.post(
        "/api/context", 
        headers={"X-API-Key": "test-key", "X-Session-ID": "test-session"},
        json={"selected_context": [safe_path]}
    )
    assert response.status_code == 200
    # Should be accepted as it is a subpath of SHARED_DOCS_DIR
    # We check if it is in the response (depending on exact abs path logic, could be modified)
    returned_context = [p.replace('\\', '/') for p in response.json()["selected_context"]]
    assert any(safe_path in p for p in returned_context)


def test_update_context_path_traversal(app_module, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/context", 
        headers={"X-API-Key": "test-key", "X-Session-ID": "test-session"},
        json={"selected_context": ["/etc/passwd"]}
    )
    assert response.status_code == 200
    # Should reject the dangerous path and fallback to SHARED_DOCS_DIR
    from src.constants import SHARED_DOCS_DIR
    assert response.json()["selected_context"] == [SHARED_DOCS_DIR]


def test_hardware_state_endpoint(app_module, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = TestClient(app_module.app)
    payload = {"temperature": 25.5, "motion_detected": True}
    response = client.post(
        "/api/hardware/state", 
        headers={"X-API-Key": "test-key"},
        json=payload
    )
    assert response.status_code == 200
    assert response.json()["state"]["temperature"] == 25.5
    assert response.json()["state"]["motion_detected"] is True


def test_langgraph_nodes_available():
    # Verify the graph nodes can be imported and are available
    from src.graph import plan_query, retrieve_hybrid, generate_grounded, update_memory
    
    assert callable(plan_query)
    assert callable(retrieve_hybrid)
    assert callable(generate_grounded)
    assert callable(update_memory)


def test_retrieve_hybrid_executes_with_fakes(app_module):
    from src.graph import retrieve_hybrid
    
    state = {
        "question": "What database are we using?",
        "chat_history": [],
        "search_queries": ["What database are we using?"],
        "documents": [],
        "generation": "",
        "hallucination_count": 0,
        "user_profile": [],
        "kg_context": "",
        "query_plan": {"retrieval_modes": ["dense"]}
    }
    
    # Run the hybrid retrieve node directly
    result = retrieve_hybrid(state)
    assert "documents" in result
    assert isinstance(result["documents"], list)


def test_update_memory_executes_with_fakes(app_module):
    from src.graph import update_memory
    
    state = {
        "question": "What database are we using?",
        "chat_history": [],
        "search_queries": [],
        "documents": [],
        "generation": "We are using Qdrant.",
        "hallucination_count": 0,
        "user_profile": ["User likes Qdrant"],
        "kg_context": "",
    }
    
    # Run the update_memory node directly
    result = update_memory(state)
    assert "user_profile" in result
    assert isinstance(result["user_profile"], list)


def test_is_safe_path():
    from src.services import is_safe_path
    from src.constants import SHARED_DOCS_DIR
    import os
    
    assert is_safe_path(SHARED_DOCS_DIR) is True
    assert is_safe_path(os.path.join(SHARED_DOCS_DIR, "file.txt")) is True
    assert is_safe_path("/etc/passwd") is False
    assert is_safe_path(os.path.join(SHARED_DOCS_DIR, "../../etc/passwd")) is False


def test_plan_query_json_fallback(monkeypatch):
    from src.graph import plan_query
    
    # Mock LLM to raise JSONDecodeError
    class MockLLM:
        def invoke(self, *args, **kwargs):
            class MockResponse:
                content = "Invalid JSON"
            return MockResponse()
            
        def bind_tools(self, *args, **kwargs):
            return self
            
    monkeypatch.setattr("src.graph.get_chat_llm", lambda **kwargs: MockLLM())
    
    state = {
        "question": "test question",
        "chat_history": []
    }
    
    result = plan_query(state)
    assert "query_plan" in result
    assert result["query_plan"]["intent"] == "lookup"
