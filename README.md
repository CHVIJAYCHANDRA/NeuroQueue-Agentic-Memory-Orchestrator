# NeuroQueue: Agentic Memory Orchestrator

An Agentic AI System with FIFO Context Memory built on Ollama + CrewAI

## Problem Solved

Traditional GPT-like models forget context or rely on cloud APIs. NeuroQueue solves this by introducing a **local, multi-agent reasoning system** that **retains contextual order** and **processes tasks sequentially like a human workflow.**

## Prerequisites

- Install Ollama: https://ollama.com/docs
- Pull the model: `ollama pull llama3:latest` (or any Ollama model)
- Python 3.10+ installed

## Setup

1. Create virtual environment and install deps:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` (optional) to set `FIFO_SIZE` or `LLM_MODEL`.

## Run CLI

```bash
python main.py
```

## Run UI (optional)

```bash
streamlit run ui.py
```

## Features

* Agentic Architecture (Planner, Analyst, Writer, Listener)
* FIFO Memory Queue for temporal reasoning with salience-based prioritization
* RAG (Retrieval-Augmented Generation) with FAISS for long-term document storage
* Parallel agent execution with consensus voting
* Local LLM (LLaMA3 via Ollama) — no external API keys
* Streamlit UI with dynamic agent outputs
* Fully offline execution

## How it works

* `fifo_memory.py` holds the short-term FIFO buffer with salience scoring for intelligent eviction.
* `rag_store.py` provides FAISS-based vector storage for long-term document retrieval.
* `ollama_client.py` issues local `ollama run <model>` calls.
* `agents.py` defines four agents: Listener, Planner, Analyst (with RAG integration), Writer.
* `main.py` orchestrates agents (sequentially or in parallel) and prints outputs + memory JSON.

## Use Cases

* Local AI Assistants with memory
* Research-grade reasoning systems
* Intelligent automation workflows
* R&D for cognitive AI design

## Configuration

### Environment Variables (`.env` file)

* `FIFO_SIZE`: Maximum size of FIFO memory buffer (default: 5)
* `LLM_MODEL`: Ollama model to use (default: `llama3:latest`)
* `ENABLE_SALIENCE`: Enable salience-based memory prioritization (default: `true`)
* `PARALLEL_MODE`: Run agents in parallel mode (default: `false`)

### UI Controls

In the Streamlit UI, you can:
* Toggle parallel execution mode
* Enable/disable RAG
* Adjust FIFO memory size dynamically
* Clear memory

## Advanced Features

### RAG (Retrieval-Augmented Generation)

The AnalystAgent automatically retrieves relevant documents from the RAG store using semantic search. Agent outputs are automatically indexed for future retrieval. The RAG store persists to `.rag_store/` directory.

### Salience Scoring

Important memories are protected from FIFO eviction through automatic salience scoring based on:
* Keyword detection (important, critical, key, etc.)
* Content length
* Manual salience assignment

### Parallel Execution & Consensus

When parallel mode is enabled, all agents execute simultaneously using ThreadPoolExecutor. A consensus mechanism synthesizes agent outputs into a unified response using the LLM.

## Notes / Troubleshooting

* If `ollama` CLI is not found: install from https://ollama.com
* To use a different model, pull it with `ollama pull <model-name>` and set `LLM_MODEL` in `.env`.
* If FAISS installation fails, the system falls back to numpy-based similarity search.
* RAG store initializes on first use — first run may be slower due to model download.
* This system is intentionally modular — you can replace call_ollama_prompt with a streaming client or LangChain Ollama wrapper later.
