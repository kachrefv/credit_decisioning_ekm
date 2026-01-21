# Credithos

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-production_ready-success)
![AI-Powered](https://img.shields.io/badge/AI-DeepSeek_Integrated-magenta)

**The Enterprise-Grade Episodic Knowledge Mesh for Credit Intelligence**

> *Credithos goes beyond traditional Rule Engines. It uses an Agentic Memory Mesh to "remember" every financial decision, grounding AI reasoning in historical precedent.*

---

## 🚀 Key Innovations

*   **Episodic Knowledge Mesh (EKM):** Unlike stateless LLM calls, Credithos maintains a persistent memory of `Borrowers`, `Applications`, and `Decisions`.
*   **Agentic Reasoning:** Integrates **DeepSeek-R1** via a RAG (Retrieval-Augmented Generation) pipeline to analyze risk factors against a vector database of similar historical cases.
*   **Incremental Learning:** The system gets smarter with every decision. New data is automatically ingested, embedded, and added to the mesh without requiring full model retraining.
*   **Clean Architecture:** Built with strict adherence to Domain-Driven Design (DDD) principles, separating Core Logic, Infrastructure, and API layers.

---

## 🛠️ Technical Stack

### Backend (The Brain)
*   **Framework:** FastAPI (Async/High Performance)
*   **AI/LLM:** DeepSeek via OpenAI Compatibility Layer
*   **Memory/Vector DB:** Qdrant Client (In-memory/Persisted) & Scikit-learn (KNN Graphs)
*   **Graph Theory:** NetworkX for risk propagation analysis
*   **Database:** PostgreSQL (SQLAlchemy ORM)

### Frontend (The Control Center)
*   **Framework:** React 19 + TypeScript + Vite
*   **Platform:** Electron (Desktop Native Experience)
*   **UI System:** TailwindCSS + HeroUI
*   **Visualization:** Interactive Data Dashboards

---

## 🏗️ Architecture Highlights

This project follows a **Fractal Clean Architecture**:

```text
src/
├── ekm/
│   ├── api/            # Primary Adapters (FastAPI)
│   ├── core/           # Kernel & Shared Logic
│   ├── domain/         # Pure Business Logic (Entities, Value Objects)
│   ├── infra/          # Secondary Adapters (DB, DeepSeek, FileSystem)
│   └── services/       # Application Services & Orchestration
```

### Pro Tip: "Cold Start" vs. "Mesh Mode"
The system is designed with a bi-modal engine:
1.  **Cold Start:** Relies on heuristic baselines when data is scarce.
2.  **Mesh Mode:** Automatically activates when the `mesh_threshold` (1000 nodes) is reached, unlocking vector-based similar case retrieval for AI decision grounding.

---

## ⚡ Quick Start

### Prerequisites
*   Python 3.11+
*   Node.js 20+

### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv311
.\venv311\Scripts\activate

# Install dependencies
pip install -e .

# Configure Environment
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### 2. Frontend Setup
```bash
cd ui
npm install
```

### 3. Run the System
```bash
# Terminal 1: Start the API Brain
python run_api.py

# Terminal 2: Launch the Electron Dashboard
cd ui && npm run dev:electron
```

---

## 👥 Authors & Maintainers

Architected with ❤️ by **Achref Riahi** and **Eya Marzougui**

*Credithos by EKM*
