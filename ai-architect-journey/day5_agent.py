from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── TOOLS ──────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Calculator result: {result}"
    except:
        return "Calculator error — invalid expression"

def get_current_datetime(_=None) -> str:
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %d %B %Y — %I:%M %p')}"

def search_knowledge(query: str) -> str:
    knowledge = {
        "groq": "Groq is a free ultra-fast LLM API. Used on Day 1.",
        "rag": "RAG is Retrieval Augmented Generation. Built on Day 4.",
        "chatbot": "Memory chatbot was built on Day 2.",
        "persistent": "Persistent memory tutor was built on Day 3.",
        "salary": "AI Architect salary goal: Year 1-2: 12-25 LPA, Year 5-7: 70-120 LPA, Year 8-10: 1.5 CR+",
        "agent": "AI Agent uses tools to take actions. Built on Day 5.",
        "langchain": "LangChain is a framework for building AI apps and agents.",
        "pytorch": "PyTorch is a deep learning framework for building neural networks."
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No specific knowledge found. Try asking differently."

def save_note(note: str) -> str:
    with open("agent_notes.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}\n")
    return f"Note saved: {note}"

# ── TOOL REGISTRY ───────────────────────────────────

tools = {
    "calculator": {
        "description": "Performs math calculations. Input: math expression like '25 * 4'",
        "function": calculator
    },
    "datetime": {
        "description": "Gets current date and time. Input: any string",
        "function": get_current_datetime
    },
    "search_knowledge": {
        "description": "Searches AI learning knowledge base. Input: topic to search",
        "function": search_knowledge
    },
    "save_note": {
        "description": "Saves important notes to file. Input: the note to save",
        "function": save_note
    }
}

# ── AGENT BRAIN ─────────────────────────────────────

def agent_decide(user_input: str) -> dict:
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in tools.items()
    ])

    prompt = f"""You are an AI Agent with these tools:
{tool_descriptions}

User message: {user_input}

Decide if you need a tool or can answer directly.
Respond ONLY in this JSON format:
{{
  "use_tool": true or false,
  "tool_name": "tool name or null",
  "tool_input": "input for tool or null",
  "direct_answer": "answer if no tool needed or null"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    
    # Extract JSON
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])

def run_agent(user_input: str) -> str:
    print(f"\n🧠 Agent thinking...")
    
    decision = agent_decide(user_input)
    
    if decision.get("use_tool"):
        tool_name = decision["tool_name"]
        tool_input = decision["tool_input"]
        
        print(f"🔧 Using tool: {tool_name}")
        print(f"📥 Tool input: {tool_input}")
        
        tool_result = tools[tool_name]["function"](tool_input)
        print(f"📤 Tool result: {tool_result}")
        
        # Final answer using tool result
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"Tool result: {tool_result}"},
                {"role": "user", "content": "Now give me a clear final answer based on this."}
            ]
        )
        return final_response.choices[0].message.content
    else:
        return decision.get("direct_answer", "I could not process that request.")

# ── MAIN ────────────────────────────────────────────

print("🤖 AI Agent — Day 5")
print("===================")
print("Available tools: calculator, datetime, search_knowledge, save_note\n")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Agent shutting down. See you on Day 6!")
        break
    
    answer = run_agent(user_input)
    print(f"\n🤖 Agent: {answer}\n")
