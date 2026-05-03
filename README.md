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

## Why I Built This

I was learning RAG and needed a real use case, not another chatbot demo. I was working on a research project at GMU analyzing 40+ papers on dating app privacy, and manually searching through them for specific findings was taking hours.

So I built this. What started as a learning project became something I kept pushing further. Basic RAG failed on cross-paper comparisons, follow-up questions, and specific technical details like exact file paths or version numbers. Each failure led to a new upgrade. By the end, the system hit 93.75% accuracy on a custom 16-question evaluation suite, handles multi-turn conversations, and is live for anyone to use right now.
This is that system.

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
│   Scores relevance - not just similarity                 │
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

I built a custom 16-question test suite with ground-truth answers spanning four difficulty levels, from single-paper factual lookups to multi-paper reasoning tasks. Running it revealed something interesting: the very hard questions outperformed the hard ones. Two of the hard category failures turned out to be evaluation design bugs, not system failures. One keyword mismatch ("triangulation" vs "trilateration"), one answer that contained the right information but missed the exact keyword. The one genuine system failure was the abstract chunk ranking higher than the experiments section for a specific app names query.

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
├── 01_load_and_chunk.py     # Pipeline step 1: PDF loading and chunking
├── 02_embed_and_store.py    # Pipeline step 2: Embeddings and vector storage
├── 03_query.py              # Pipeline step 3: Basic RAG query pipeline
│
├── Dockerfile               # Container config for Hugging Face deployment
├── .streamlit/config.toml   # Streamlit server configuration
├── .env                     # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Known Limitations & Future Work

**Current limitations:**
- PDF tables don't extract cleanly. Data stored in tables (like vulnerability comparison tables in the Kim paper) gets missed or garbled by PyPDF. PyMuPDF would handle this better.
- The vector store resets on every restart. Deliberate tradeoff to avoid a Windows file locking issue where ChromaDB held database files open and crashed when reprocessing papers.
- Architecture starts degrading around 15 papers. Per-paper retrieval gets slow and routing becomes less reliable as topics overlap. Scaling to 30+ papers requires a paper-level summary index as a first retrieval stage.
- Evaluation uses keyword matching as a proxy for correctness. Two questions were marked wrong due to keyword mismatches, not actual system failures. RAGAS or LLM-as-judge scoring would be more precise.

**Planned improvements:**
- [ ] PyMuPDF integration for accurate table extraction
- [ ] Persistent vector store with proper session management
- [ ] Paper-level summary index for 30+ paper collections
- [ ] LLM-as-judge evaluation using RAGAS framework
- [ ] Fine-tuned domain-specific embedding model
- [ ] Multi-document iterative retrieval for cross-paper inventory questions


---

## What I Learned

The biggest gap between a RAG tutorial and a real system is that tutorials stop at the point where things start getting hard.

**Chunking strategy matters as much as the model.** Started with 500-character chunks, retrieval quality was visibly worse. Moving to 1000 with overlap and switching to RecursiveCharacterTextSplitter improved results before touching any retrieval logic.

**Retrieval failures and generation failures need different fixes.** When the system gave a wrong answer, the instinct was to blame the LLM. Half the time it was the wrong chunks being retrieved upstream. Distinguishing the two made debugging dramatically faster.

**Hybrid search is worth the complexity.** Adding BM25 alongside vector search specifically helped with exact technical strings like file paths, version numbers, and field names that semantic search consistently missed.

**Evaluation design is its own engineering problem.** Two test questions failed due to keyword mismatches in my own evaluation suite, not system failures. Building the eval revealed as much about the system as building the system did.

---

## Author

**Rohith Reddy Kar**
MS in Applied Information Technology (Machine Learning) - George Mason University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/rohith-reddy-kar-76536226a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Rohith2320)

---

<div align="center">

*If this project helped you, consider giving it a ⭐*

</div>
