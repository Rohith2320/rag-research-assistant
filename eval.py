"""
Automated evaluation suite for the Research Paper QA Bot.
Tests retrieval accuracy and answer quality across 16 questions.

Usage: python eval.py
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
import tempfile

# ============================================================
# GROUND TRUTH EVALUATION SET
# 16 questions across 4 difficulty levels
# ============================================================

EVAL_QUESTIONS = [
    # --- EASY: Single paper, answer stated directly ---
    {
        "id": "E1",
        "difficulty": "Easy",
        "question": "Which five specific apps were selected and analyzed in the Kim et al. experiments - Tinder, Amanda, Noondate, Glam, and DangYeonsi?",
        "target_paper": "When_Harry_Met_Tinder_Security_Analysis.pdf",
        "required_keywords": ["tinder", "amanda", "noondate", "glam", "dangyeonsi"],
        "ground_truth": "Tinder, Amanda, Noondate, Glam, and DangYeonsi"
    },
    {
        "id": "E2",
        "difficulty": "Easy",
        "question": "What device and Android version were used in the Farnden et al. forensic analysis experiments?",
        "target_paper": "Privacy Risks in Mobile Dating Apps.pdf",
        "required_keywords": ["samsung", "galaxy", "4.1.2"],
        "ground_truth": "Samsung Galaxy S3 GT-I9300T running Android 4.1.2"
    },
    {
        "id": "E3",
        "difficulty": "Easy",
        "question": "How many email addresses were extracted during the profile collection attack in the Kim study and how long did it take?",
        "target_paper": "When_Harry_Met_Tinder_Security_Analysis.pdf",
        "required_keywords": ["883", "121"],
        "ground_truth": "883 email addresses were extracted in 121.81 seconds"
    },
    {
        "id": "E4",
        "difficulty": "Easy",
        "question": "What is the path where Grindr stores its database on Android devices according to the Farnden paper?",
        "target_paper": "Privacy Risks in Mobile Dating Apps.pdf",
        "required_keywords": ["com.grindapp.android"],
        "ground_truth": "/data/data/com.grindapp.android"
    },
    # --- MEDIUM: Requires finding specific detail ---
    {
        "id": "M1",
        "difficulty": "Medium",
        "question": "Which dating apps in the Shetty et al. study did NOT achieve any of the three Goal States?",
        "target_paper": "Are_You_Dating_Danger_An_Interdisciplinary_Approach_to_Evaluating_the_InSecurity_of_Android_Dating_Apps.pdf",
        "required_keywords": ["lovoo", "coffee"],
        "ground_truth": "Lovoo and Coffee Meets Bagel did not achieve any of the three Goal States"
    },
    {
        "id": "M2",
        "difficulty": "Medium",
        "question": "How did the researchers bypass SSL pinning in the Kim et al. study?",
        "target_paper": "When_Harry_Met_Tinder_Security_Analysis.pdf",
        "required_keywords": ["xposed", "ssl", "unpinning"],
        "ground_truth": "Using ssl_unpinning.apk provided by the Xposed framework"
    },
    {
        "id": "M3",
        "difficulty": "Medium",
        "question": "What specific Facebook user data could an adversary retrieve using a stolen Tinder access token according to the Shetty paper?",
        "target_paper": "Are_You_Dating_Danger_An_Interdisciplinary_Approach_to_Evaluating_the_InSecurity_of_Android_Dating_Apps.pdf",
        "required_keywords": ["birthday", "photos", "friends", "education"],
        "ground_truth": "user_birthday, user_relationship_details, user_likes, user_education_history, user_work_history, user_photos, user_friends, public_profile"
    },
    {
        "id": "M4",
        "difficulty": "Medium",
        "question": "How many Google Plus recommendations and members does Meet Me have according to the Farnden paper?",
        "target_paper": "Privacy Risks in Mobile Dating Apps.pdf",
        "required_keywords": ["145", "90 million"],
        "ground_truth": "145 thousand Google+ recommendations and 90 million members"
    },
    # --- HARD: Requires cross-paper synthesis ---
    {
        "id": "H1",
        "difficulty": "Hard",
        "question": "What common attack method is discussed across all the uploaded papers as a threat to dating app users?",
        "target_paper": None,
        "required_keywords": ["mitm", "man-in-the-middle"],
        "ground_truth": "Man-in-the-Middle (MITM) attacks are discussed across all papers"
    },
    {
        "id": "H2",
        "difficulty": "Hard",
        "question": "Compare the methodology used in the Farnden paper with the approach used in the Kim paper.",
        "target_paper": None,
        "required_keywords": ["forensic", "packet", "storage", "farnden", "kim"],
        "ground_truth": "Farnden used forensic case study approach; Kim used packet analysis, API hooking, storage analysis, and code decompilation"
    },
    {
        "id": "H3",
        "difficulty": "Hard",
        "question": "Across all papers, which specific apps were found to store or leak location data and how?",
        "target_paper": None,
        "required_keywords": ["tinder", "grindr", "location"],
        "ground_truth": "Tinder, Grindr, Skout leak location data through various mechanisms including GPS coordinates and triangulation"
    },
    {
        "id": "H4",
        "difficulty": "Hard",
        "question": "What countermeasures or recommendations do the papers collectively suggest to improve dating app security?",
        "target_paper": None,
        "required_keywords": ["encryption", "ssl", "user education"],
        "ground_truth": "Multiple countermeasures including SSL/TLS improvements, encryption, user education, high-entropy indexes, vague location data"
    },
    # --- VERY HARD: Multi-step reasoning ---
    {
        "id": "VH1",
        "difficulty": "Very Hard",
        "question": "Which co-located attacker findings in the Kim paper are confirmed or extended by the Farnden forensic analysis?",
        "target_paper": None,
        "required_keywords": ["credential", "shared", "storage"],
        "ground_truth": "Both papers find credentials stored in shared preferences and chat messages recoverable from storage"
    },
    {
        "id": "VH2",
        "difficulty": "Very Hard",
        "question": "Based on all papers, what is the most dangerous combination of vulnerabilities an attacker could exploit against a dating app user?",
        "target_paper": None,
        "required_keywords": ["location", "credential", "profile"],
        "ground_truth": "Combination of location tracking, credential theft, and profile enumeration enables stalking, identity theft, and physical harm"
    },
    {
        "id": "VH3",
        "difficulty": "Very Hard",
        "question": "How do the privacy concerns identified in the Stoicescu paper relate to the actual vulnerabilities demonstrated in the Shetty and Kim papers?",
        "target_paper": None,
        "required_keywords": ["stoicescu", "shetty", "kim"],
        "ground_truth": "Stoicescu identifies theoretical concerns about location, credentials, and data sharing which Shetty and Kim demonstrate as exploitable vulnerabilities"
    },
    {
        "id": "VH4",
        "difficulty": "Very Hard",
        "question": "If a malicious actor wanted to track a specific dating app user's physical location, which techniques from across all papers would be most effective?",
        "target_paper": None,
        "required_keywords": ["trilateration", "location", "distance"],
        "ground_truth": "Triangulation using distance information (Kim), exact GPS from Grindr (Farnden), and location inference from proximity (Stoicescu)"
    }
]


# ============================================================
# RAG SYSTEM (mirrors app.py logic)
# ============================================================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def clean_text(text):
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text

def build_system(pdf_paths):
    """Build the complete RAG system from PDF files."""
    print("Building RAG system...")
    all_chunks = []
    paper_info = {}

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"  Loading: {filename}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_file"] = filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

        # Extract paper info from first page
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Extract title and authors from this text:
{pages[0].page_content[:1000]}
Format: TITLE: ... AUTHORS: ..."""
        response = llm.invoke(prompt)
        paper_info[filename] = response.content

    print(f"  Total chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    tokenized = [doc.page_content.lower().split() for doc in all_chunks]
    bm25_index = BM25Okapi(tokenized)

    return vectorstore, bm25_index, all_chunks, paper_info


def hybrid_search(question, vectorstore, bm25_index, bm25_chunks,
                  k=10, filter_paper=None):
    """Hybrid vector + BM25 search with RRF fusion."""
    if filter_paper:
        vector_results = vectorstore.similarity_search(
            question, k=k, filter={"source_file": filter_paper}
        )
    else:
        vector_results = vectorstore.similarity_search(question, k=k)

    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    if filter_paper:
        for i, chunk in enumerate(bm25_chunks):
            if chunk.metadata.get("source_file") != filter_paper:
                bm25_scores[i] = 0

    top_bm25_indices = sorted(range(len(bm25_scores)),
                               key=lambda i: bm25_scores[i],
                               reverse=True)[:k]
    bm25_results = [bm25_chunks[i] for i in top_bm25_indices
                    if bm25_scores[i] > 0]

    rrf_scores = {}
    for rank, doc in enumerate(vector_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)

    seen = set()
    combined = []
    for doc in sorted(vector_results + bm25_results,
                       key=lambda d: rrf_scores.get(d.page_content[:100], 0),
                       reverse=True):
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    return combined[:k]


def route_query(question, paper_names, paper_info=None):
    """Route query to specific paper or all papers."""
    comparative_signals = [
        "compare", "both", "all papers", "across", "each paper",
        "difference", "similar", "contrast", "versus", "vs",
        "collectively", "which paper", "multiple", "all studies",
        "same vulnerability", "same finding", "relate", "confirm",
        "extend", "combination", "all uploaded"
    ]

    if any(signal in question.lower() for signal in comparative_signals):
        return None

    if paper_info:
        paper_list = "\n".join([
            f"- Filename: {name} | {paper_info.get(name, '')}"
            for name in paper_names
        ])
    else:
        paper_list = "\n".join([f"- {name}" for name in paper_names])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(f"""Given this question and paper list, return ONLY the 
filename if the question targets a specific paper, or "ALL" if general.

Papers:
{paper_list}

Question: {question}

Response (filename or ALL):""")

    result = response.content.strip()
    return result if result in paper_names else None


def answer_question(question, vectorstore, bm25_index, bm25_chunks,
                    paper_names, paper_info):
    """Full RAG pipeline for a single question."""
    target_paper = route_query(question, paper_names, paper_info)

    if target_paper:
        results = hybrid_search(question, vectorstore, bm25_index,
                                bm25_chunks, k=20, filter_paper=target_paper)
    else:
        results = []
        chunks_per_paper = max(4, 16 // len(paper_names))
        for paper_name in paper_names:
            try:
                paper_results = hybrid_search(
                    question, vectorstore, bm25_index, bm25_chunks,
                    k=chunks_per_paper, filter_paper=paper_name
                )
                results.extend(paper_results)
            except Exception:
                continue

    if len(results) > 1:
        scores = reranker.predict(
            [(question, doc.page_content) for doc in results]
        )
        results = [doc for _, doc in sorted(
            zip(scores, results), key=lambda x: x[0], reverse=True
        )[:8]]

    context = "\n\n".join([
        f"[From {doc.metadata.get('source_file', '?')}, "
        f"page {doc.metadata.get('page', '?')}]:\n{doc.page_content}"
        for doc in results
    ])

    paper_list = "\n".join([f"- {name}" for name in paper_names])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke(f"""You are a research assistant. Answer based ONLY on 
the context below. Be specific and cite sources.

Papers available:
{paper_list}

Context:
{context}

Question: {question}

Answer:""")

    return response.content, target_paper


# ============================================================
# EVALUATION LOGIC
# ============================================================

def evaluate_answer(answer, eval_item):
    """Score an answer against ground truth using keyword matching."""
    answer_lower = answer.lower()
    
    keywords_found = sum(
        1 for kw in eval_item["required_keywords"]
        if kw.lower() in answer_lower
    )
    keyword_score = keywords_found / len(eval_item["required_keywords"])
    
    # Score: 1.0 = all keywords found, 0.5 = half, 0.0 = none
    if keyword_score >= 0.75:
        verdict = "PASS"
    elif keyword_score >= 0.4:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    
    return verdict, keyword_score, keywords_found, len(eval_item["required_keywords"])


def run_evaluation(pdf_paths):
    """Run the full evaluation suite."""
    print("\n" + "="*60)
    print("RAG EVALUATION SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Build system
    vectorstore, bm25_index, bm25_chunks, paper_info = build_system(pdf_paths)
    paper_names = list(paper_info.keys())

    results = []
    total_time = 0

    print(f"\nRunning {len(EVAL_QUESTIONS)} evaluation questions...\n")

    for i, eval_item in enumerate(EVAL_QUESTIONS):
        print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {eval_item['id']}: {eval_item['question'][:60]}...")

        start = time.time()
        try:
            answer, routed_to = answer_question(
                eval_item["question"], vectorstore, bm25_index,
                bm25_chunks, paper_names, paper_info
            )
            elapsed = time.time() - start
            total_time += elapsed

            verdict, score, found, total_kw = evaluate_answer(answer, eval_item)

            results.append({
                "id": eval_item["id"],
                "difficulty": eval_item["difficulty"],
                "question": eval_item["question"],
                "answer": answer,
                "verdict": verdict,
                "keyword_score": score,
                "keywords_found": found,
                "total_keywords": total_kw,
                "routed_to": routed_to or "ALL",
                "time_seconds": round(elapsed, 2)
            })

            status_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
            print(f"  {status_icon} {verdict} ({found}/{total_kw} keywords) — {elapsed:.1f}s")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                "id": eval_item["id"],
                "difficulty": eval_item["difficulty"],
                "question": eval_item["question"],
                "answer": f"ERROR: {e}",
                "verdict": "ERROR",
                "keyword_score": 0,
                "keywords_found": 0,
                "total_keywords": len(eval_item["required_keywords"]),
                "routed_to": "N/A",
                "time_seconds": 0
            })

    # ============================================================
    # REPORT
    # ============================================================
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"pass": 0, "partial": 0, "fail": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if r["verdict"] == "PASS":
            by_difficulty[d]["pass"] += 1
        elif r["verdict"] == "PARTIAL":
            by_difficulty[d]["partial"] += 1
        else:
            by_difficulty[d]["fail"] += 1

    total_pass = sum(1 for r in results if r["verdict"] == "PASS")
    total_partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
    total_fail = sum(1 for r in results if r["verdict"] in ["FAIL", "ERROR"])
    total_questions = len(results)

    print(f"\nOverall Score: {total_pass}/{total_questions} PASS "
          f"({total_partial} PARTIAL, {total_fail} FAIL)")
    print(f"Accuracy: {total_pass/total_questions*100:.1f}% "
          f"(+partial: {(total_pass+total_partial*0.5)/total_questions*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s "
          f"(avg: {total_time/total_questions:.1f}s per question)")

    print("\nBy difficulty:")
    for difficulty in ["Easy", "Medium", "Hard", "Very Hard"]:
        if difficulty in by_difficulty:
            d = by_difficulty[difficulty]
            print(f"  {difficulty}: {d['pass']}/{d['total']} pass, "
                  f"{d['partial']} partial, {d['fail']} fail")

    print("\nDetailed results:")
    print(f"{'ID':<5} {'Difficulty':<10} {'Verdict':<8} {'Score':<8} {'Routed To':<15} {'Time'}")
    print("-"*65)
    for r in results:
        routed = r['routed_to'][:14] if r['routed_to'] != 'ALL' else 'ALL papers'
        print(f"{r['id']:<5} {r['difficulty']:<10} {r['verdict']:<8} "
              f"{r['keyword_score']:.2f}    {routed:<15} {r['time_seconds']}s")

    # Save results to JSON
    output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total_questions,
                "pass": total_pass,
                "partial": total_partial,
                "fail": total_fail,
                "accuracy": total_pass/total_questions,
                "avg_time": total_time/total_questions
            },
            "results": results
        }, f, indent=2)

    print(f"\nFull results saved to: {output_file}")
    return results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Point these to your actual PDF files
    PDF_PATHS = [
        "./When_Harry_Met_Tinder_Security_Analysis.pdf",
        "./Are_You_Dating_Danger_An_Interdisciplinary_Approach_to_Evaluating_the_InSecurity_of_Android_Dating_Apps.pdf",
        "./Privacy Risks in Mobile Dating Apps.pdf",
        "./Sharing_and_Privacy_in_Dating_Apps.pdf"
    ]

    # Check files exist
    missing = [p for p in PDF_PATHS if not os.path.exists(p)]
    if missing:
        print("Missing PDF files:")
        for m in missing:
            print(f"  {m}")
        print("\nMake sure your PDFs are in the project folder.")
        sys.exit(1)

    run_evaluation(PDF_PATHS)