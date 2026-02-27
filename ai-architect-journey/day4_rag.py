from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 1 — Your document (paste any text here)
document = """
AI Architect Learning Journey

Day 1: Learned Groq API, prompt engineering, and built first AI script.
Day 2: Built a memory chatbot using conversation history and system prompts.
Day 3: Built a persistent AI tutor that saves and loads sessions using JSON.
Day 4: Learning RAG — Retrieval Augmented Generation.

Key Skills for AI Architect:
- Python programming
- Groq and OpenAI APIs
- LangChain for building AI apps
- PyTorch for deep learning
- AWS/GCP for cloud deployment

Salary Goals:
- Year 1-2: 12-25 LPA
- Year 3-4: 30-60 LPA  
- Year 5-7: 70-120 LPA
- Year 8-10: 1.5 CR+
"""

# Step 2 — Split document into chunks
def split_into_chunks(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3 — Find relevant chunks for a question
def find_relevant_chunks(question, chunks, top_n=2):
    question_words = set(question.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_n]]

# Step 4 — Answer using relevant chunks
def ask_document(question, chunks):
    relevant = find_relevant_chunks(question, chunks)
    context = "\n\n".join(relevant)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    return response.choices[0].message.content

# Main program
print("📚 RAG Document Q&A System")
print("==========================\n")

chunks = split_into_chunks(document)
print(f"✅ Document loaded and split into {len(chunks)} chunks\n")
print("Type 'quit' to exit\n")

while True:
    question = input("Your Question: ")
    
    if question.lower() == "quit":
        print("Goodbye!")
        break
    
    answer = ask_document(question, chunks)
    print(f"\n🤖 Answer: {answer}\n")
