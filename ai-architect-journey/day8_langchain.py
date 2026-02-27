from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. LLM SETUP ────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

print("=" * 50)
print("LangChain Day 8 - Three Power Features")
print("=" * 50)

# ── 2. FEATURE 1: PROMPT TEMPLATES ──────────────────
print("\n📌 Feature 1: Prompt Templates")
print("-" * 30)

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}. Answer in exactly {lines} lines."),
    ("human", "{question}")
])

chain = template | llm

response = chain.invoke({
    "topic": "AI Architecture",
    "lines": "3",
    "question": "What is the most important skill for an AI Architect?"
})

print(f"Q: What is the most important skill for an AI Architect?")
print(f"A: {response.content}\n")

# ── 3. FEATURE 2: CONVERSATION MEMORY ───────────────
print("📌 Feature 2: Built-in Conversation Memory")
print("-" * 30)

# Modern memory using a simple list
chat_history = []

memory_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Architect tutor."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def chat_with_memory(user_input):
    chain = memory_template | llm
    response = chain.invoke({
        "history": chat_history,
        "input": user_input
    })
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    return response.content

r1 = chat_with_memory("My name is AI Architect Student")
print(f"Turn 1: {r1[:100]}...\n")

r2 = chat_with_memory("What is my name?")
print(f"Turn 2 (memory test): {r2}\n")

# ── 4. FEATURE 3: CHAIN CHAINING ────────────────────
print("📌 Feature 3: Chaining Prompts Together")
print("-" * 30)

concept_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI expert. Explain concepts simply."),
    ("human", "Explain {concept} in one sentence.")
])

example_template = ChatPromptTemplate.from_messages([
    ("system", "You are a practical AI teacher."),
    ("human", "Give one real world example of this: {explanation}")
])

concept_chain = concept_template | llm
example_chain = example_template | llm

concept_response = concept_chain.invoke({"concept": "Vector Embeddings"})
explanation = concept_response.content
print(f"Concept: {explanation}\n")

example_response = example_chain.invoke({"explanation": explanation})
print(f"Real World Example: {example_response.content}\n")

print("=" * 50)
print("✅ Day 8 Complete - LangChain Mastered!")
print("=" * 50)
