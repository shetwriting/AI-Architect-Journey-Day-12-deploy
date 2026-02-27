# -*- coding: utf-8 -*-
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

print("Day 10 -- Fine-tuning LLMs")
print("=" * 50)

# -- PART 1: THREE APPROACHES --

print("\n[Part 1] Three Customization Approaches")
print("-" * 40)

approaches = {
    "Prompt Engineering": {
        "cost": "Free",
        "speed": "Instant",
        "power": "Low-Medium",
        "use_when": "Quick customization, general tasks",
        "example": "You are an AI Architect tutor..."
    },
    "RAG": {
        "cost": "Low",
        "speed": "Fast",
        "power": "Medium-High",
        "use_when": "Large knowledge bases, frequently updated data",
        "example": "Search company docs -> inject into prompt"
    },
    "Fine-tuning": {
        "cost": "High",
        "speed": "Days/Weeks",
        "power": "Highest",
        "use_when": "Specific style, domain expertise, private data",
        "example": "Train on 10,000 medical Q&A pairs"
    }
}

for name, details in approaches.items():
    print(f"\n>> {name}")
    for key, value in details.items():
        print(f"   {key}: {value}")

# -- PART 2: BUILD A FINE-TUNING DATASET --

print("\n\n[Part 2] Creating Fine-tuning Dataset")
print("-" * 40)
print("Generating AI Architect training data...\n")

training_data = []

topics = [
    "What is a vector database?",
    "Explain RAG in simple terms",
    "What is LangChain used for?",
    "How do AI Agents work?",
]

template = ChatPromptTemplate.from_messages([
    ("system", """You are creating training data for an AI Architect tutor model.
Generate a high quality answer that is:
- Clear and educational
- Practical and example-based
- Encouraging to learners
- Under 150 words"""),
    ("human", "Question: {question}\n\nProvide a training answer:")
])

chain = template | llm

for topic in topics:
    response = chain.invoke({"question": topic})
    pair = {
        "instruction": "You are an AI Architect tutor. Answer clearly and practically.",
        "input": topic,
        "output": response.content
    }
    training_data.append(pair)
    print(f"[OK] Generated: {topic[:50]}...")

with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"\n[SAVED] {len(training_data)} training pairs to training_data.json")

# -- PART 3: GENERIC VS FINE-TUNED COMPARISON --

print("\n\n[Part 3] Generic vs Fine-tuned Model Behavior")
print("-" * 40)

generic_template = ChatPromptTemplate.from_messages([
    ("system", "You are a general AI assistant."),
    ("human", "{question}")
])

specialized_template = ChatPromptTemplate.from_messages([
    ("system", """You are an elite AI Architect tutor specialized in training
students from India to reach 1CR+ salary. You know their journey:
- They started with zero knowledge
- Built 10 projects in 10 days
- Current skills: Groq, LangChain, RAG, Agents, Vector DB, Fine-tuning
- Goal: AI Architect role in 3-5 years
Always relate answers to their specific journey and Indian tech market."""),
    ("human", "{question}")
])

question = "How long will it take me to get a good job in AI?"

generic_chain = generic_template | llm
specialized_chain = specialized_template | llm

print(f"\nQuestion: {question}")

print("\n[GENERIC MODEL ANSWER]:")
generic_response = generic_chain.invoke({"question": question})
print(generic_response.content[:400])
print()

print("[FINE-TUNED MODEL ANSWER]:")
specialized_response = specialized_chain.invoke({"question": question})
print(specialized_response.content[:400])

# -- PART 4: DECISION FRAMEWORK --

print("\n\n[Part 4] Fine-tune vs RAG vs Prompting Decision Framework")
print("-" * 40)

scenarios = [
    ("Customer support bot for your company", "RAG"),
    ("Medical diagnosis assistant", "Fine-tuning"),
    ("General Q&A chatbot", "Prompt Engineering"),
    ("Legal document analyzer", "Fine-tuning + RAG"),
    ("Personal AI tutor", "RAG + Prompt Engineering"),
    ("Code completion for specific codebase", "Fine-tuning"),
    ("News summarizer", "Prompt Engineering"),
    ("Company policy assistant", "RAG"),
]

print("\nScenario -> Best Approach:")
for scenario, approach in scenarios:
    print(f"  > {scenario}")
    print(f"    -> {approach}\n")

# -- PART 5: SAVE FINE-TUNING GUIDE --

guide = {
    "fine_tuning_steps": [
        "1. Collect domain-specific data (minimum 500-1000 examples)",
        "2. Format as instruction-input-output pairs",
        "3. Choose base model (LLaMA, Mistral, Phi)",
        "4. Use LoRA/QLoRA for efficient fine-tuning",
        "5. Train on GPU (Google Colab free tier works)",
        "6. Evaluate on test set",
        "7. Deploy fine-tuned model"
    ],
    "recommended_tools": [
        "Unsloth - fastest fine-tuning library",
        "Axolotl - flexible training framework",
        "HuggingFace - model hub and training",
        "Google Colab - free GPU for training",
        "Weights & Biases - experiment tracking"
    ],
    "training_data_sources": [
        "Manual creation (highest quality)",
        "GPT-4 generated synthetic data",
        "Web scraping domain content",
        "Company internal documents",
        "Public datasets on HuggingFace"
    ]
}

with open("finetune_guide.json", "w", encoding="utf-8") as f:
    json.dump(guide, f, indent=2, ensure_ascii=False)

print("[SAVED] Fine-tuning guide saved to finetune_guide.json")

print("\n" + "=" * 50)
print("[DONE] Day 10 Complete -- Fine-tuning Mastered!")
print("=" * 50)
