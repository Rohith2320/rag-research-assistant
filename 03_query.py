from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
load_dotenv()

# Connect to the existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print(f"Connected to ChromaDB with {vectorstore._collection.count()} chunks")

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Ask a question
query = "Can the application Frida be misused?"

# Retrieve relevant chunks
results = vectorstore.similarity_search(query, k=10)

# Build the prompt with retrieved context
context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""You are a research assistant. Answer the question based ONLY on the 
following context. If the context doesn't contain the answer, say "I don't have 
enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

# Send to LLM
response = llm.invoke(prompt)
print(f"\nQuestion: {query}")
print(f"\nAnswer: {response.content}")




