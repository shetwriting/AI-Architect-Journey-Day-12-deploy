from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ── 1. YOUR KNOWLEDGE BASE ───────────────────────────
documents = [
    "Day 1: Built first Groq API script. Learned prompt engineering and .env file setup.",
    "Day 2: Built memory chatbot using conversation history and system prompts.",
    "Day 3: Built persistent AI tutor that saves sessions using JSON file storage.",
    "Day 4: Built RAG system using keyword matching to answer from documents.",
    "Day 5: Built AI Agent with 4 tools — calculator, datetime, search, save note.",
    "Day 6: Built Multi-Agent pipeline with Researcher, Writer, Critic, Manager agents.",
    "Day 7: Built Full Stack Web App with elegant dark UI running in browser.",
    "Day 8: Learned LangChain — prompt templates, conversation memory, chain chaining.",
    "AI Architect salary: Year 1-2 is 12-25 LPA, Year 3-4 is 30-60 LPA, Year 5-7 is 70-120 LPA, Year 8-10 is 1.5 CR+",
    "Key AI Architect skills: Python, Groq API, LangChain, PyTorch, AWS, Vector Databases, RAG systems, AI Agents.",
    "Vector embeddings convert text into numbers so computers can find similar meanings.",
    "ChromaDB is a free open source vector database perfect for local AI development.",
    "RAG stands for Retrieval Augmented Generation — AI that reads your own documents.",
    "LangChain is the industry standard framework used in 80% of production AI apps.",
    "Multi-agent systems use specialized agents — each with one job — to solve complex problems.",
]

# ── 2. CREATE VECTOR DATABASE ────────────────────────
print("🚀 Day 9 — Vector Database + Advanced RAG")
print("=" * 50)
print("\n📦 Loading embedding model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔢 Creating vector embeddings...")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

print(f"✅ {len(documents)} documents stored as vectors!\n")

# ── 3. SEMANTIC SEARCH FUNCTION ──────────────────────
def semantic_search(query, n_results=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # Cosine similarity
    dot_products = np.dot(doc_embeddings, query_embedding.T).flatten()
    norms = np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = dot_products / (norms + 1e-10)

    top_indices = np.argsort(similarities)[::-1][:n_results]
    return [documents[i] for i in top_indices]

# ── 4. ADVANCED RAG FUNCTION ─────────────────────────
def advanced_rag(question):
    relevant_docs = semantic_search(question)
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    template = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Architect tutor. 
Answer based on the context provided.
Be specific, encouraging, and relate to the student's journey.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = template | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, relevant_docs

# ── 5. TEST IT ───────────────────────────────────────
print("🔍 Testing Semantic Search")
print("-" * 30)

test_queries = [
    "What have I learned about memory?",
    "How much money will I make?",
    "What tools did my agent have?",
    "How do I store AI knowledge?",
]

for query in test_queries:
    print(f"\n❓ Query: {query}")
    answer, sources = advanced_rag(query)
    print(f"📚 Sources found: {len(sources)}")
    print(f"🤖 Answer: {answer[:200]}...")
    print()

# ── 6. INTERACTIVE MODE ──────────────────────────────
print("=" * 50)
print("💬 Interactive Advanced RAG — Ask anything!")
print("Type 'quit' to exit\n")

while True:
    question = input("Your Question: ")
    if question.lower() == "quit":
        print("See you on Day 10!")
        break

    answer, sources = advanced_rag(question)
    print(f"\n📚 Retrieved {len(sources)} relevant documents")
    print(f"\n🤖 Answer: {answer}\n")
    print("-" * 40)
