from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import re

load_dotenv()

#step 1: laod
loader = PyPDFLoader("test.pdf")
pages = loader.load()

#step 2: clean
def clean_text(text):
    # Fix hyphenated line breaks (e.g., "locatio-\nns" -> "locations")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Replace single newlines with spaces (preserve paragraph breaks)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    return text
for page in pages:
    page.page_content = clean_text(page.page_content)

#step 3: Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)
print(f"created {len(chunks)} chunks")

#step 4: Embed and store in ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Stored {len(chunks)} chunks in ChromaDB at ./chroma_db")

#Step 5: test a search
query = "What security vulnerabilities were found in dating apps?"
results = vectorstore.similarity_search(query, k=10)

print(f"\n--- Top 3 results for: '{query}' ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1} (Page {doc.metadata.get('page', '?')}):")
    print(doc.page_content[:200])