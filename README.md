---
title: Rag Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

<div align="center">

# 🔬 Research Paper QA Bot

### Ask questions across multiple academic papers. Get cited, accurate answers.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Eval%20Accuracy-93.75%25-2ea44f?style=flat)](#evaluation)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

<br/>

> > **Live Demo:** [🚀 Try it on Hugging Face Spaces](https://huggingface.co/spaces/Rohith2320/rag-research-assistant)

<br/>

![Demo placeholder](https://via.placeholder.com/800x400/1a1a2e/ffffff?text=Research+Paper+QA+Bot+Demo)

</div>

---

## The Problem

Reading 10 research papers to find a specific finding takes hours. Basic RAG systems help — but they fail on questions that require comparing across papers, understanding follow-up context, or finding exact technical details like file paths, version numbers, and numeric results.

**This project solves that** with a production-grade, multi-stage retrieval pipeline that outperforms standard RAG on cross-paper synthesis and specific factual queries.

---

## What Makes This Different

Most RAG tutorials use a single vector search. This system uses **five layered retrieval strategies**:

| Standard RAG | This System |
|---|---|
| One search across all chunks | Per-paper search + result fusion |
| Vector similarity only | Hybrid: Vector + BM25 keyword |
| No query understanding | LLM-based query routing + reformulation |
| Static retrieval | Cross-encoder re-ranking |
| No conversation context | 4-turn conversation memory |
| Manual testing | Automated 16-question eval suite |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Question                        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Query Reformulation                         │
│   Expands pronouns using 4-turn conversation history     │
│   "Which apps did they find?" → "Which apps did          │
│    Shetty et al. find vulnerable to MITM attacks?"       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Query Router                           │
│   Detects comparative signals ("compare", "across",      │
│   "both") → routes to ALL papers                         │
│   Detects paper references ("Farnden", "Kim paper") →   │
│   routes to specific paper only                          │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌──────────────────┐           ┌──────────────────────────┐
│ Single-paper     │           │ Multi-paper search        │
│ Hybrid Search    │           │ Per-paper Hybrid Search   │
│ (metadata filter)│           │ × N papers in parallel    │
└──────────┬───────┘           └─────────────┬────────────┘
           │                                 │
           └──────────────┬──────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Hybrid Search (per paper)                   │
│                                                          │
│   Vector Search (OpenAI embeddings)                      │
│         +                                                │
│   BM25 Keyword Search (exact term matching)              │
│         ↓                                                │
│   Reciprocal Rank Fusion (RRF)                           │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           Cross-encoder Re-ranking                       │
│   ms-marco-MiniLM reads (question, chunk) together       │
│   Scores relevance — not just similarity                 │
│   Reduces 20 candidates → top 5-8                        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              GPT-4o-mini Generation                      │
│   Context labeled with source file + page number         │
│   Instructed to attribute claims to specific papers      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│         Cited Answer + Source Pages + Search Scope       │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

Evaluated against a custom 16-question test suite with ground-truth answers across four difficulty levels. Questions range from single-paper factual lookups to multi-paper reasoning tasks.

<div align="center">

| Difficulty | Questions | Score | Accuracy |
|:---:|:---:|:---:|:---:|
| 🟢 Easy | 4 | 4/4 | 100% |
| 🟡 Medium | 4 | 4/4 | 100% |
| 🟠 Hard | 4 | 3/4 | 75% |
| 🔴 Very Hard | 4 | 4/4 | 100% |
| **Total** | **16** | **15/16** | **93.75%** |

**Average response time: 6.4 seconds per query**

</div>

### Example Q&A

**Question:** *"Compare the methodology used in the Farnden paper with the approach used in the Kim paper."*

**Answer:**
> The Farnden paper employed a forensic case study methodology, simulating user actions on 9 proximity-based dating apps and recovering artefacts from device storage (Privacy Risks in Mobile Dating Apps.pdf, p.0). In contrast, the Kim paper used a dual static/dynamic analysis framework comprising packet analysis, API hooking, storage analysis, and code decompilation to identify software vulnerabilities in 5 Android dating apps (When_Harry_Met_Tinder_Security_Analysis.pdf, p.4, p.6).

*🔍 Searched: All 4 papers (4 chunks each) → re-ranked to top 8*

---

## Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|---|---|---|
| **Embeddings** | `text-embedding-3-small` | Semantic vector representation |
| **Generation** | `GPT-4o-mini` | Answer synthesis |
| **Re-ranking** | `ms-marco-MiniLM-L-6-v2` | Precision relevance scoring |
| **Vector DB** | ChromaDB | Embedding storage + similarity search |
| **Keyword Search** | BM25 (rank-bm25) | Exact term matching |
| **Framework** | LangChain | Pipeline orchestration |
| **UI** | Streamlit | Web interface |
| **Evaluation** | Custom Python suite | Automated accuracy testing |

</div>

---

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com))
- One or more PDF research papers

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rohith2320/rag-research-assistant.git
cd rag-research-assistant

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install langchain langchain-openai langchain-community langchain-chroma \
            chromadb openai streamlit pypdf tiktoken python-dotenv \
            langchain-text-splitters sentence-transformers rank-bm25

# 4. Configure environment
echo OPENAI_API_KEY=your-api-key-here > .env

# 5. Launch the app
streamlit run app.py
```

### Usage

1. Open `http://localhost:8501` in your browser
2. Upload one or more PDF papers using the sidebar
3. Wait for processing (~15-30 seconds per paper)
4. Ask questions in the chat interface

### Running the Evaluation Suite

```bash
# Place your PDFs in the project directory, then:
python eval.py

# Results saved to eval_results_[timestamp].json
```

---

## Project Structure

```
rag-research-assistant/
│
├── app.py                   # Main Streamlit application
│   ├── clean_text()         # PDF text normalization
│   ├── extract_paper_info() # LLM-based metadata extraction
│   ├── build_bm25_index()   # Keyword search index builder
│   ├── hybrid_search()      # Vector + BM25 + RRF fusion
│   ├── route_query()        # LLM + signal-based query routing
│   ├── reformulate_query()  # Conversation-aware query expansion
│   └── ask_question()       # Full RAG pipeline orchestrator
│
├── eval.py                  # Automated 16-question evaluation suite
│
├── 01_load_and_chunk.py     # Learning: PDF loading and chunking
├── 02_embed_and_store.py    # Learning: Embeddings and vector storage
├── 03_query.py              # Learning: Basic RAG query pipeline
│
├── .env                     # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Known Limitations & Future Work

**Current limitations:**
- PDF tables are not extracted accurately — structured data in tables may be missed
- In-memory vector store resets on restart (tradeoff to avoid file locking on Windows)
- Architecture scales to ~10-15 papers; 30+ papers requires paper-level summary indexing
- Keyword-based evaluation is a proxy — LLM-as-judge scoring would be more precise

**Planned improvements:**
- [ ] PyMuPDF integration for accurate table extraction
- [ ] Persistent vector store with proper session management
- [ ] Paper-level summary index for 30+ paper collections
- [ ] LLM-as-judge evaluation using RAGAS framework
- [ ] Fine-tuned domain-specific embedding model

---

## What I Learned

Building this project taught me the practical gap between "RAG tutorial" and "production RAG system":

- **Chunking strategy** matters as much as the model — fixed-size chunking vs. semantic chunking produces measurably different retrieval quality
- **Retrieval failures and generation failures** are distinct failure modes requiring different fixes
- **Hybrid search** consistently outperforms pure vector search on domain-specific technical content
- **Evaluation design** is as hard as system design — keyword matching revealed its own failure modes

---

## Author

**Rohith Reddy Kar**
MS in Applied Information Technology (Machine Learning) — George Mason University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/rohith-reddy-kar-76536226a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Rohith2320)

---

<div align="center">

*If this project helped you, consider giving it a ⭐*

</div>
