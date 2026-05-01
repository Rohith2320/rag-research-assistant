from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

loader = PyPDFLoader("test.pdf")
pages = loader.load()

def clean_text(text):
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<=[a-z])(?:\s*)\n(?:\s*)(?=[a-z])', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text

for page in pages:
    page.page_content = clean_text(page.page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print(f"\n--- Chunking Results ---")
print(f"Original pages: {len(pages)}")
print(f"After chunking: {len(chunks)}")
print(f"\n--- Chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Chunk 2 ---")
print(chunks[1].page_content)

for i, chunk in enumerate(chunks):
    if "locatio" in chunk.page_content:
        print(f"\n--- Chunk {i} raw ---")
        print(repr(chunk.page_content))
        break