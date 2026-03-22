import importlib.util
import sys
import types
from pathlib import Path

import pytest


class _SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def container(self):
        return self

    def markdown(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None


class _Placeholder(_NoOpContext):
    pass


class _FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.sidebar = _NoOpContext()

    def set_page_config(self, *args, **kwargs):
        return None

    def cache_resource(self, func=None, **kwargs):
        if func is None:
            def decorator(inner):
                return inner
            return decorator
        return func

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def link_button(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def multiselect(self, *args, **kwargs):
        return []

    def toggle(self, label, key=None, **kwargs):
        if key is not None and key not in self.session_state:
            self.session_state[key] = False
        return self.session_state.get(key, False)

    def spinner(self, *args, **kwargs):
        return _NoOpContext()

    def chat_message(self, *args, **kwargs):
        return _NoOpContext()

    def expander(self, *args, **kwargs):
        return _NoOpContext()

    def empty(self):
        return _Placeholder()

    def chat_input(self, *args, **kwargs):
        return None

    def write_stream(self, generator):
        return "".join(list(generator))

    def rerun(self):
        return None


class _FakePhoenix(types.ModuleType):
    def __init__(self):
        super().__init__("phoenix")

    def launch_app(self):
        return object()


class _FakePhoenixOtel(types.ModuleType):
    def __init__(self):
        super().__init__("phoenix.otel")

    def register(self):
        return object()


class _FakeInstrumentor:
    def instrument(self, *args, **kwargs):
        return None


class _FakeOpenInference(types.ModuleType):
    def __init__(self):
        super().__init__("openinference.instrumentation.langchain")

    class LangChainInstrumentor(_FakeInstrumentor):
        pass


class _FakeObserver:
    def schedule(self, *args, **kwargs):
        return None

    def start(self):
        return None


class _FakeFileSystemEventHandler:
    pass


class _FakeWatchdogEvents(types.ModuleType):
    def __init__(self):
        super().__init__("watchdog.events")
        self.FileSystemEventHandler = _FakeFileSystemEventHandler


class _FakeWatchdogPolling(types.ModuleType):
    def __init__(self):
        super().__init__("watchdog.observers.polling")
        self.PollingObserver = _FakeObserver


class _FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _BaseLoader:
    def __init__(self, file_path, *args, **kwargs):
        self.file_path = file_path

    def load(self):
        return [_FakeDocument(f"Loaded content from {self.file_path}", {"source": self.file_path})]


class _FakeDocumentLoaders(types.ModuleType):
    def __init__(self):
        super().__init__("langchain_community.document_loaders")
        self.PyMuPDFLoader = _BaseLoader
        self.TextLoader = _BaseLoader
        self.Docx2txtLoader = _BaseLoader
        self.UnstructuredMarkdownLoader = _BaseLoader


class _FakeTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs):
        return docs


class _FakeTextSplitters(types.ModuleType):
    def __init__(self):
        super().__init__("langchain_text_splitters")
        self.RecursiveCharacterTextSplitter = _FakeTextSplitter


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeChatOllama:
    def __init__(self, model=None, base_url=None, temperature=0, format=None):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.format = format

    def invoke(self, prompt):
        if isinstance(prompt, list):
            prompt = "\n".join(getattr(item, "content", str(item)) for item in prompt)

        if "Generate 3 distinct search queries" in prompt:
            return _FakeLLMResponse(
                "requirements.txt contents\nvector database used\ndocker services overview"
            )

        if '"score": "yes"' in prompt or '"score": "no"' in prompt or "Rules:" in prompt:
            return _FakeLLMResponse('{"score": "yes"}')

        if "Extract permanent facts" in prompt or "Extract durable user facts" in prompt:
            return _FakeLLMResponse(
                '{"facts": ["User is testing CI"], "triples": [["User", "is", "testing CI"]]}'
            )

        return _FakeLLMResponse(
            "Qdrant is the vector database used by the application. Source: architecture.md"
        )

    async def ainvoke(self, prompt):
        return self.invoke(prompt)


class _FakeOllamaEmbeddings:
    def __init__(self, model=None, base_url=None):
        self.model = model
        self.base_url = base_url


class _FakeRetriever:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def invoke(self, query):
        if self.collection_name == "chat_history":
            return [
                _FakeDocument(
                    "Past note: the system has discussed Qdrant before.",
                    {"source": "Past Conversation Memory"},
                )
            ]

        if "requirements" in query.lower():
            return [
                _FakeDocument(
                    "requirements.txt includes langchain, qdrant-client, streamlit, and arize-phoenix.",
                    {"source": "requirements.txt"},
                )
            ]

        if "database" in query.lower() or "vector" in query.lower():
            return [
                _FakeDocument(
                    "The application uses Qdrant as a vector database in a Docker container.",
                    {"source": "architecture.md"},
                )
            ]

        return [
            _FakeDocument(
                "General app context mentioning Qdrant and Streamlit.",
                {"source": "overview.md"},
            )
        ]


class _FakeQdrantVectorStore:
    def __init__(self, client=None, collection_name=None, embedding=None):
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding

    @classmethod
    def from_documents(cls, docs, embeddings, url=None, collection_name=None):
        return cls(client=None, collection_name=collection_name, embedding=embeddings)

    def as_retriever(self, search_kwargs=None):
        return _FakeRetriever(self.collection_name)

    def add_texts(self, texts, metadatas=None):
        return None


class _FakeLangchainOllama(types.ModuleType):
    def __init__(self):
        super().__init__("langchain_ollama")
        self.ChatOllama = _FakeChatOllama
        self.OllamaEmbeddings = _FakeOllamaEmbeddings


class _FakeLangchainQdrant(types.ModuleType):
    def __init__(self):
        super().__init__("langchain_qdrant")
        self.QdrantVectorStore = _FakeQdrantVectorStore


class _FakeCollectionInfo:
    def __init__(self, points_count=0):
        self.points_count = points_count


class _FakeQdrantClient:
    _collections = {}

    def __init__(self, url=None):
        self.url = url

    def collection_exists(self, collection_name):
        return collection_name in self._collections

    def create_collection(self, collection_name, vectors_config=None):
        self._collections[collection_name] = {"points_count": 0}

    def delete_collection(self, collection_name):
        self._collections.pop(collection_name, None)

    def delete(self, collection_name=None, points_selector=None):
        return None

    def get_collection(self, collection_name):
        data = self._collections.get(collection_name, {"points_count": 0})
        return _FakeCollectionInfo(points_count=data["points_count"])

    def scroll(self, collection_name=None, scroll_filter=None, limit=1):
        return [], None


class _FakeDistance:
    COSINE = "cosine"


class _FakeVectorParams:
    def __init__(self, size=None, distance=None):
        self.size = size
        self.distance = distance


class _FakeMatchValue:
    def __init__(self, value=None):
        self.value = value


class _FakeMatchAny:
    def __init__(self, any=None):
        self.any = any


class _FakeFieldCondition:
    def __init__(self, key=None, match=None):
        self.key = key
        self.match = match


class _FakeFilter:
    def __init__(self, must=None):
        self.must = must or []


class _FakeQdrantRest(types.ModuleType):
    def __init__(self):
        super().__init__("qdrant_client.http.models")
        self.Distance = _FakeDistance
        self.VectorParams = _FakeVectorParams
        self.MatchValue = _FakeMatchValue
        self.MatchAny = _FakeMatchAny
        self.FieldCondition = _FakeFieldCondition
        self.Filter = _FakeFilter


class _FakeQdrantClientModule(types.ModuleType):
    def __init__(self):
        super().__init__("qdrant_client")
        self.QdrantClient = _FakeQdrantClient


class _FakeQdrantHttpModule(types.ModuleType):
    def __init__(self, rest_module):
        super().__init__("qdrant_client.http")
        self.models = rest_module


class _FakeRanker:
    def __init__(self, model_name=None, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir

    def rerank(self, rerank_request):
        ranked = []
        for passage in rerank_request.passages:
            ranked.append(
                {
                    "text": passage["text"],
                    "meta": passage["meta"],
                    "score": 1.0,
                }
            )
        return ranked


class _FakeRerankRequest:
    def __init__(self, query=None, passages=None):
        self.query = query
        self.passages = passages or []


class _FakeFlashrank(types.ModuleType):
    def __init__(self):
        super().__init__("flashrank")
        self.Ranker = _FakeRanker
        self.RerankRequest = _FakeRerankRequest


class _FakeCompiledGraph:
    def invoke(self, state, config=None):
        return {
            "question": state.get("question", ""),
            "search_queries": state.get("search_queries", []),
            "documents": state.get("documents", []),
            "generation": state.get("generation", ""),
            "hallucination_count": state.get("hallucination_count", 0),
            "user_profile": state.get("user_profile", []),
            "kg_context": state.get("kg_context", ""),
        }

    def get_state(self, config=None):
        return types.SimpleNamespace(values={})


class _FakeStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type

    def add_node(self, *args, **kwargs):
        return None

    def set_entry_point(self, *args, **kwargs):
        return None

    def add_edge(self, *args, **kwargs):
        return None

    def add_conditional_edges(self, *args, **kwargs):
        return None

    def compile(self, checkpointer=None):
        return _FakeCompiledGraph()


class _FakeLanggraphGraph(types.ModuleType):
    def __init__(self):
        super().__init__("langgraph.graph")
        self.StateGraph = _FakeStateGraph
        self.END = "END"


class _FakePostgresSaver:
    def __init__(self, conn):
        self.conn = conn

    def setup(self):
        pass


class _FakeLanggraphCheckpointPostgres(types.ModuleType):
    def __init__(self):
        super().__init__("langgraph.checkpoint.postgres")
        self.PostgresSaver = _FakePostgresSaver


class _FakeConnectionPool:
    def __init__(self, *args, **kwargs):
        pass

    def connection(self):
        return _NoOpContext()

    def open(self):
        pass


class _FakePsycopgPool(types.ModuleType):
    def __init__(self):
        super().__init__("psycopg_pool")
        self.ConnectionPool = _FakeConnectionPool


class _FakeGraph:
    def __init__(self):
        self.nodes = []
        self._edges = []

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, source, target, relation=None):
        self.add_node(source)
        self.add_node(target)
        self._edges.append((source, target, {"relation": relation}))

    def edges(self, node=None, data=False):
        if node is None:
            items = list(self._edges)
        else:
            items = [edge for edge in self._edges if edge[0] == node]
        if data:
            return items
        return [(src, dst) for src, dst, _ in items]


class _FakeNetworkX(types.ModuleType):
    def __init__(self):
        super().__init__("networkx")
        self._graphs = {}

    def DiGraph(self):
        return _FakeGraph()

    def write_graphml(self, graph, path):
        self._graphs[str(path)] = graph

    def read_graphml(self, path):
        return self._graphs.get(str(path), _FakeGraph())


def _install_fake_modules():
    fake_phoenix = _FakePhoenix()
    fake_phoenix_otel = _FakePhoenixOtel()
    fake_openinference = _FakeOpenInference()
    fake_watchdog_events = _FakeWatchdogEvents()
    fake_watchdog_polling = _FakeWatchdogPolling()
    fake_doc_loaders = _FakeDocumentLoaders()
    fake_splitters = _FakeTextSplitters()
    fake_ollama = _FakeLangchainOllama()
    fake_qdrant_store = _FakeLangchainQdrant()
    fake_rest = _FakeQdrantRest()
    fake_qdrant_client = _FakeQdrantClientModule()
    fake_qdrant_http = _FakeQdrantHttpModule(fake_rest)
    fake_flashrank = _FakeFlashrank()
    fake_langgraph_graph = _FakeLanggraphGraph()
    fake_langgraph_checkpoint_postgres = _FakeLanggraphCheckpointPostgres()
    fake_psycopg_pool = _FakePsycopgPool()
    fake_networkx = _FakeNetworkX()

    sys.modules["phoenix"] = fake_phoenix
    sys.modules["phoenix.otel"] = fake_phoenix_otel
    sys.modules["openinference.instrumentation.langchain"] = fake_openinference
    sys.modules["watchdog.events"] = fake_watchdog_events
    sys.modules["watchdog.observers.polling"] = fake_watchdog_polling
    sys.modules["langchain_community.document_loaders"] = fake_doc_loaders
    sys.modules["langchain_text_splitters"] = fake_splitters
    sys.modules["langchain_ollama"] = fake_ollama
    sys.modules["langchain_qdrant"] = fake_qdrant_store
    sys.modules["qdrant_client"] = fake_qdrant_client
    sys.modules["qdrant_client.http"] = fake_qdrant_http
    sys.modules["qdrant_client.http.models"] = fake_rest
    sys.modules["flashrank"] = fake_flashrank
    sys.modules["langgraph.graph"] = fake_langgraph_graph
    sys.modules["langgraph.checkpoint.postgres"] = fake_langgraph_checkpoint_postgres
    sys.modules["psycopg_pool"] = fake_psycopg_pool
    sys.modules["networkx"] = fake_networkx


def _load_app_module():
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    candidate_paths = [
        repo_root / "app.py",
        repo_root / "src" / "app.py",
        repo_root / "server.py",
        repo_root / "src" / "server.py",
    ]

    app_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            app_path = candidate
            break

    if app_path is None:
        raise FileNotFoundError("Could not find app.py or server.py in repository root or src/")

    spec = importlib.util.spec_from_file_location("app_under_test", app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["app_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def app_module():
    _install_fake_modules()
    return _load_app_module()


@pytest.fixture
def fake_doc_class():
    return _FakeDocument