# ⚡ FinSight AI: Enterprise-Grade Financial RAG Pipeline

An end-to-end, production-oriented Retrieval-Augmented Generation (RAG) system designed for deterministic data extraction and analytical reasoning over complex financial reports (Balance Sheets, Income Statements, Cash Flow Statements).

This repository documents the architectural evolution from a standard document-QA script to a decoupled **FastAPI + Streamlit** microservices architecture, utilizing isolated environments to ensure data integrity and stateless deployment.

---

## 💼 Business Value & Market Positioning

General-purpose Large Language Models (LLMs) often struggle with numerical precision in financial contexts. FinSight AI is positioned as an **Automated Corporate Intelligence Tool** for the B2B sector:

* **Target Audience:** Investment Analysts, Credit Risk Officers, Audit Firms, and Corporate Strategy Teams.
* **Problem Statement:** Navigating 150-page annual reports is time-consuming and prone to human error. Locating specific non-current liabilities or niche financial footnotes requires significant manual effort.
* **Solution:** FinSight AI reduces retrieval time to seconds while anchoring the LLM's reasoning strictly to the source matrices (tables).

---

## 🏗️ System Architecture & MLOps (v2.0)

The monolithic structure has been refactored into a low-latency, stateless REST API architecture.

### Data Flow & Design

`Client (Streamlit) ➔ FastAPI ➔ LlamaParse (Matrix Generation) ➔ ChromaDB (Isolated Vaults) ➔ Dual-Brain RAG (GPT-4o-mini) ➔ Response`

* **Decoupled Architecture:** The UI functions as a "Dumb Client." All heavy-lifting processes—vectorization, LLM orchestration, and disk I/O—are handled by the isolated FastAPI backend.
* **Zero-Leakage Vaults:** Instead of a single persistent database, the system generates a unique, isolated `ChromaDB` collection (`vault_TIMESTAMP`) for each uploaded report. A garbage collection mechanism prunes old vaults to prevent data contamination between different corporate entities.
* **Containerized Ecosystem:** Orchestrated via `docker-compose`, the services communicate over an internal bridge network, ensuring a fully isolated production environment.

---

## 🔬 Failure Analysis & Technical Solutions

The pipeline incorporates specific countermeasures against three common failure modes in standard RAG architectures.

### 1. The Table Fracture Paradox
* **Challenge:** Conventional PDF parsers treat tables as unstructured text strings. Standard chunking often severs rows and columns, destroying the financial matrix.
* **Solution:** Integrated **LlamaParse** with `result_type="markdown"` to preserve tabular structures as aligned grids.
* **Implementation:** The `RecursiveCharacterTextSplitter` was re-configured with specialized separators: `separators=["\n\n", "\n", "|", " ", ""]`. The inclusion of `|` ensures that matrix rows remain intact during the chunking process.

### 2. Hallucination Mitigation: "Mental Math"
* **Challenge:** LLMs frequently attempt to calculate financial totals internally (mental math), leading to fabricated numerical data.
* **Solution:** Implemented a strict **System Prompt** architecture. The model is explicitly restricted from performing internal calculations, forced instead to follow Markdown `|` boundaries and extract pre-existing values directly from the source.

### 3. Context Bleed & Dual-Brain Architecture
* **Challenge:** In conversational RAG, follow-up queries (e.g., "What about 2024?") often fail vector searches because the query lacks the necessary financial keywords found in the document.
* **Solution:**
    1. **Translator Brain:** Evaluates conversational history and raw input to synthesize a standalone, optimized search query (e.g., "Total Revenue 2024").
    2. **Core Analyst Brain:** Processes the retrieved context using the user's *original* input, ensuring conversational continuity while maintaining mathematical precision in retrieval.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **API Framework:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Orchestration:** LangChain, OpenAI (`gpt-4o-mini`)
* **Parsing:** LlamaParse (LlamaIndex)
* **Vector DB:** ChromaDB
* **Deployment:** Docker & Docker Compose

---

## 🚀 Installation & Quick Start

The system is fully containerized. Docker is the only requirement.

**1. Clone the repository:**
```bash
git clone https://github.com/kadir-a/financial-rag-system.git
cd financial-rag-system
```

**2. Configure Environment Variables:**
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
```

**3. Build and Launch:**
```bash
docker compose up -d --build
```

**4. Access:**
* **Frontend:** `http://localhost:8501`
* **API Documentation (Swagger):** `http://localhost:8000/docs`

---

## 🚧 Limitations & Roadmap (WIP)

While functional, this pipeline is in active development to address the following technical benchmarks:

* **Analytical Reasoning:** Current reliance on `gpt-4o-mini` is sufficient for retrieval but requires upgrading for complex multi-step financial growth analysis.
* **Quantitative Execution:** The roadmap includes integrating a **Pandas Agent** or **Python REPL** to shift from text-based reading to deterministic mathematical execution over retrieved matrices.
* **Multi-Document Synthesis:** Future iterations will allow cross-referencing multiple historical reports within a single isolated vault.