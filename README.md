# LocalRAG: Autonomous Research Agent (Staff Edition)

[![CI/CD](https://img.shields.io/badge/CI%2FCD-DeepEval-green)](./tests)
[![Stack](https://img.shields.io/badge/Stack-LangGraph%20%7C%20Qdrant%20%7C%20Docker-blue)](./docker-compose.yml)
[![Infrastructure](https://img.shields.io/badge/Infrastructure-Docker%20Compose-orange)](./docker-compose.yml)

**LocalRAG** is a privacy-first, retrieval-augmented generation (RAG) platform designed for **autonomous document research**. 

Unlike standard RAG pipelines, LocalRAG implements a **cyclic agentic architecture** using **LangGraph**, allowing the system to audit its own answers, detect hallucinations, and self-correct in real-time. It runs entirely offline on consumer hardware (RTX 3090/4090) using containerized microservices.

---

## Architecture

The system follows a **Microservices Pattern** orchestrated via Docker Compose. It decouples the Inference Engine (Compute) from the State Management (Vector DB) and Application Logic.

![System Architecture](https://mermaid.ink/img/pako:eNp9Ul2P2jAQ_CsrP1QgQUQIx0dUVaLQokqnQu-gD014MMmWWCR2tHHoccB_74b0OCpdaynObrwzu57JUUQmRuGLLck8geU01MBrVSA1gmpfN-F9u_3h9GgJZZYqC6svJxjnecBPqiJpldEwMdpKpbm8xtd7UW5q2lBMTbRDgq9ofxnaQWNBai8tNkNRl1aLCaFqNacowcISnxfcaovaBvdSb2cXrku-fkXd4KuTC4PrwANaUrjHE3yLSTJDIxR1BN8xsoZg-jEUzfVb8G4Fbz9IvTvB51QWSRUG1wgmi9WbOM-BGbIIPPgJ5mkqMxnUL5gtVvCItFcR_m92h1nGZcwq3xuT_7l8XYU6_oe28w37tJcbxe4c4B0sWTylt39J-0q_JBlVui4Sg1o9BaEYk3rGl5ytMWSh3-n02ZybWStWpGBxYFfsuub6tJdpeWPTdVLR4h9KxcK3VGJLZEiZrFJxrEpCYRPMMBQ-h7GkXTXqmTG51D-MyV5gZMptIvyfMi04K_OYW02V5Htn16_E3ZAmptRW-K7buZAI_yiehN-_c7zuwBsMe163PxiNWuIg_G7fGXmD3tD13N6o67md3rklni9dO85wcHf-DUe49DM)

* **User Flow:** The User interacts with the Streamlit UI, which sends requests to the FastAPI backend.
* **Agentic Loop:** The LangGraph agent orchestrates retrieval from Qdrant, re-ranking via FlashRank, and generation via Ollama.
* **Self-Correction:** If the Hallucination Grader fails, the agent autonomously loops back to retry the generation.

---

## Key Features

### 1. **Self-Healing Agentic Loops**
Instead of a linear chain (`Retrieve -> Generate`), this system uses a **State Graph**.
* **Hallucination Grader:** After generating an answer, a secondary LLM call verifies if the claims are grounded in the retrieved context.
* **Retry Mechanism:** If a hallucination is detected, the graph loops back to the generation step with a penalty prompt.

### 2. **Two-Stage Retrieval (Hybrid Search)**
To solve the "Lost in the Middle" phenomenon:
* **Stage 1:** Broad retrieval of top 10 documents using Dense Vector Search (Cosine Similarity).
* **Stage 2:** **Re-Ranking** using a Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) running locally on CPU to filter for the top 3 semantically relevant chunks.

### 3. **Production-Grade Observability**
Integrated **Arize Phoenix** (OpenTelemetry) to trace every step of the pipeline.
* **Latency Tracing:** Visualize exactly how long Retrieval took vs. Token Generation.
* **Token Counting:** Monitor cost (simulated) and throughput.

### 4. **Automated Unit Testing (LLM-as-a-Judge)**
Implements **Test-Driven Development (TDD)** for RAG.
* Uses **DeepEval** to run regression tests before deployment.
* A local Llama-3 model acts as a "Judge" to score answers for **Faithfulness** and **Relevancy**.

---

## Tech Stack & Trade-offs

| Component | Tool Choice | Why this over the alternative? |
| :--- | :--- | :--- |
| **Inference** | **Ollama (Docker)** | Provides a stable, OpenAI-compatible API layer over raw `llama.cpp` bindings, simplifying container networking. |
| **Vector DB** | **Qdrant** | Chosen over ChromaDB for its Rust-based performance, ability to handle millions of vectors, and built-in hybrid search capabilities. |
| **Orchestration** | **LangGraph** | Chosen over standard LangChain Chains to enable **Cyclic Graphs** (Loops) required for self-correction. |
| **Observability** | **Arize Phoenix** | The only open-source, local-first OTEL collector that provides visual trace waterfalls without a cloud login. |

---

## Getting Started

### Prerequisites
* **Docker Desktop**.
* **NVIDIA GPU** (RTX 30XX or 40XX recommended) with updated drivers.
* **RAM:** 32GB+ recommended (for running Docker + Chrome + VS Code).

### 1. Spin Up the Stack
This single command launches the Database, Inference Engine, Dashboard, and UI.
```bash
docker-compose up -d