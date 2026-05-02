import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re
import os
import tempfile
load_dotenv()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Page Config ---
st.set_page_config(
    page_title="Research Paper QA",
    page_icon="🔬",
    layout="wide"
)

# --- Text Cleaning ---
def clean_text(text):
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text
#query routing
def route_query(question, paper_names, paper_info=None, chat_history=None):
    """Use LLM to determine if a question targets a specific paper."""
    
    # Force ALL papers for comparative/multi-paper questions
    comparative_signals = [
        "compare", "both", "all papers", "across", "each paper",
        "difference", "similar", "contrast", "versus", "vs",
        "all studies", "collectively", "which paper", "multiple",
        "other paper", "other study", "also find", "did they",
        "same vulnerability", "same finding", "agree", "disagree"
    ]
    
    question_lower = question.lower()
    if any(signal in question_lower for signal in comparative_signals):
        return None  # None means search ALL papers
    
    # Also check recent conversation for comparative context
    if chat_history:
        recent_questions = [
            msg['content'].lower() 
            for msg in chat_history[-3:] 
            if msg['role'] == 'user'
        ]
        if any(
            any(signal in q for signal in comparative_signals)
            for q in recent_questions
        ):
            return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if paper_info:
        paper_list = "\n".join([
            f"- Filename: {name} | {paper_info.get(name, 'No info')}" 
            for name in paper_names
        ])
    else:
        paper_list = "\n".join([f"- {name}" for name in paper_names])
    
    routing_prompt = f"""Given the following question and list of available papers, 
determine if the question is asking about a specific paper or about all papers.

Available papers:
{paper_list}

Question: {question}

Rules:
- If the question mentions a specific author name, paper title, or clear description of ONE paper → return that filename
- If the question is general, comparative, or about multiple papers → return "ALL"
- If unsure → return "ALL"

Your response must be either a single filename from the list or "ALL". Nothing else."""

    response = llm.invoke(routing_prompt)
    result = response.content.strip()
    
    if result in paper_names:
        return result
    return None

#query reformulation
def reformulate_query(question, chat_history):
    """Reformulate question using conversation history for better retrieval."""
    if not chat_history:
        return question
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    history_text = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:300]}"
        for msg in chat_history[-4:]
    ])
    
    prompt = f"""Given this conversation history and the new question, rewrite the 
question to be self-contained and specific. Replace pronouns like "they", "it", 
"this", "that" with the actual entities they refer to. Keep it concise.

Conversation history:
{history_text}

New question: {question}

Rewritten question (if already self-contained, return as-is):"""

    response = llm.invoke(prompt)
    return response.content.strip()

def extract_paper_info(pages, filename):
    """Use LLM to extract title and authors from the first page."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    first_page = pages[0].page_content[:1500]
    
    prompt = f"""From this first page of an academic paper, extract:
1. The paper title
2. The author names (comma-separated)

Text:
{first_page}

Respond in exactly this format:
TITLE: [title here]
AUTHORS: [author1, author2, author3]"""

    response = llm.invoke(prompt)
    return response.content
#bm25 index builder
def build_bm25_index(chunks):
    """Build a BM25 keyword index from document chunks."""
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    return BM25Okapi(tokenized), chunks
# hybrid search
def hybrid_search(question, vectorstore, bm25_index, bm25_chunks, 
                  k=10, filter_paper=None):
    """Combine vector search and BM25 keyword search using RRF."""
    
    # Vector search
    if filter_paper:
        vector_results = vectorstore.similarity_search(
            question, k=k, filter={"source_file": filter_paper}
        )
    else:
        vector_results = vectorstore.similarity_search(question, k=k)
    
    # BM25 keyword search
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Filter BM25 by paper if needed
    if filter_paper:
        for i, chunk in enumerate(bm25_chunks):
            if chunk.metadata.get("source_file") != filter_paper:
                bm25_scores[i] = 0
    
    top_bm25_indices = sorted(range(len(bm25_scores)), 
                               key=lambda i: bm25_scores[i], 
                               reverse=True)[:k]
    bm25_results = [bm25_chunks[i] for i in top_bm25_indices 
                    if bm25_scores[i] > 0]
    
    # Reciprocal Rank Fusion (RRF)
    # Give each chunk a score based on its rank in each list
    rrf_scores = {}
    
    for rank, doc in enumerate(vector_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
    
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
    
    # Collect unique docs sorted by RRF score
    seen = set()
    combined = []
    
    all_docs = vector_results + bm25_results
    all_docs_sorted = sorted(all_docs, 
                              key=lambda d: rrf_scores.get(d.page_content[:100], 0),
                              reverse=True)
    
    for doc in all_docs_sorted:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    
    return combined[:k] 
# --- Process PDFs ---
def process_pdfs(uploaded_files):
    all_chunks = []
    paper_info = {}

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

         # Extract paper metadata from first page
        info = extract_paper_info(pages, uploaded_file.name)
        paper_info[uploaded_file.name] = info

        # Add filename to metadata so we know which paper each chunk came from
        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_file"] = uploaded_file.name
            page.metadata["paper_info"] = info


        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

        os.unlink(tmp_path)

    # Clear old database if it exists
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
    )
    # bm25 keyword index

    bm25_index, bm25_chunks = build_bm25_index(all_chunks)

    return vectorstore, len(all_chunks), paper_info, bm25_index, bm25_chunks

# --- Query Function ---
def ask_question(vectorstore, question, paper_names, chat_history=None):
    # Step 1: Reformulate query using conversation history
    if chat_history:
        search_query = reformulate_query(question, chat_history)
    else:
        search_query = question

    # Step 2: Route using reformulated query
    target_paper = route_query(search_query, paper_names, 
                               st.session_state.get("paper_info"), chat_history)

    bm25_index = st.session_state.get("bm25_index")
    bm25_chunks = st.session_state.get("bm25_chunks", [])

    # Step 3: Hybrid retrieval 
    if target_paper:
        if bm25_index:
            results = hybrid_search(search_query, vectorstore, bm25_index,
                                   bm25_chunks, k=20, filter_paper=target_paper)
        else:
            results = vectorstore.similarity_search(search_query, k=20,
                                                   filter={"source_file": target_paper})
        search_scope = f"Searched: {target_paper}"
    else:
        results = []
        chunks_per_paper = max(4, 16 // len(paper_names))
        for paper_name in paper_names:
            try:
                if bm25_index:
                    paper_results = hybrid_search(search_query, vectorstore,
                                                 bm25_index, bm25_chunks,
                                                 k=chunks_per_paper,
                                                 filter_paper=paper_name)
                else:
                    paper_results = vectorstore.similarity_search(
                        search_query, k=chunks_per_paper,
                        filter={"source_file": paper_name})
                results.extend(paper_results)
            except Exception:
                continue
        search_scope = f"Searched: All {len(paper_names)} papers ({chunks_per_paper} chunks each)"

    # Step 4: Re-rank
    if len(results) > 1:
        passages = [doc.page_content for doc in results]
        scores = reranker.predict([(search_query, passage) for passage in passages])
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_k = 5 if target_paper else min(8, len(results))
        results = [doc for _, doc in scored_results[:top_k]]
        search_scope += f" → re-ranked to top {len(results)}"

    # Step 5: Build context
    context_parts = []
    sources = []
    for doc in results:
        page = doc.metadata.get('page', '?')
        filename = doc.metadata.get('source_file', 'Unknown')
        context_parts.append(f"[From {filename}, page {page}]:\n{doc.page_content}")
        sources.append(f"{filename} (p.{page})")

    context = "\n\n".join(context_parts)
    paper_list = "\n".join([f"- {name}" for name in paper_names])

    # Step 6: Build conversation history for prompt
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:500]}"
            for msg in recent
        ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are a research assistant helping scholars analyze academic papers.
The following papers have been uploaded:
{paper_list}

{"Previous conversation:" + chr(10) + history_text + chr(10) if history_text else ""}
Answer the question based ONLY on the following context from these papers. Each chunk 
is labeled with its source file and page number. When comparing across papers, clearly 
state which finding comes from which paper. If this is a follow-up question, use the 
conversation history to understand references to previous answers.

Context:
{context}

Current question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, list(dict.fromkeys(sources)), search_scope
# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "paper_names" not in st.session_state:
    st.session_state.paper_names = []
if "paper_info" not in st.session_state:
    st.session_state.paper_info = {}
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "bm25_chunks" not in st.session_state:
    st.session_state.bm25_chunks = []

# --- Sidebar ---
with st.sidebar:
    st.header("🔬 Research Paper QA")
    st.caption("Upload papers and ask questions about their content.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]

        # Only reprocess if new files are uploaded
        if file_names != st.session_state.paper_names:
            with st.spinner("Reading and processing your papers..."):
                try:
                
                    vectorstore, num_chunks, paper_info, bm25_index, bm25_chunks = process_pdfs(uploaded_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.paper_names = file_names
                    st.session_state.paper_info = paper_info
                    st.session_state.bm25_index = bm25_index
                    st.session_state.bm25_chunks = bm25_chunks
                    st.session_state.messages = []
                    st.success(f"Processed {len(uploaded_files)} paper(s) into {num_chunks} chunks.")
                except Exception as e:
                    st.error(f"Failed to process papers: {str(e)}")
                    st.info("Make sure yoyr files are valid PDFs and OpenAI API key has credits.")
    if st.session_state.paper_names:
        st.divider()
        st.subheader("Loaded Papers")
        for name in st.session_state.paper_names:
            st.write(f"📄 {name}")

    st.divider()
    if st.button("🗑️ Clear everything"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.caption("Built with LangChain, ChromaDB & OpenAI")

# --- Main Chat Area ---
if st.session_state.vectorstore:

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"📎 Sources: {', '.join(msg['sources'])}")
            if msg.get("scope"):
                st.caption(f"🔍 {msg['scope']}")

    # Chat input
    if question := st.chat_input("Ask a question about your papers..."):

        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources, scope = ask_question(
                        st.session_state.vectorstore,
                        question,
                        st.session_state.paper_names,
                        chat_history=st.session_state.messages
                    )
                    st.write(answer)
                    st.caption(f"📎 Sources: {', '.join(sources)}")
                    st.caption(f"🔍 {scope}")
                except Exception as e:
                    answer = f"Sorry, I encountered an error: {str(e)}"
                    sources = []
                    scope = ""
                    st.error(answer)
                    st.info("This may be due to API rate limits. Please try again.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "scope": scope
        })

else:
    st.markdown("### Welcome to Research Paper QA Bot 👋")
    st.markdown("Upload one or more academic research papers in the sidebar to get started.")

    st.divider()

    st.markdown("**How it works:**")
    st.markdown("""
1. Upload PDFs using the sidebar (research papers, reports, or any academic documents)
2. Wait ~20-30 seconds for processing
3. Ask questions in plain English
4. Get cited answers with page references
    """)

    st.divider()

    st.markdown("**Example questions you can ask:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- What methodology did the authors use?
- What were the key findings?
- What limitations did they mention?
        """)
    with col2:
        st.markdown("""
- Compare the approaches across papers
- What countermeasures do they suggest?
- Which specific apps were analyzed?
        """)

    st.divider()
    st.caption("Built with LangChain · ChromaDB · OpenAI · Streamlit")