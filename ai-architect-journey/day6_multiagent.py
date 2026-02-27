from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── AGENT FACTORY ───────────────────────────────────

def create_agent(name: str, role: str, goal: str):
    return {
        "name": name,
        "system_prompt": f"""You are {name}.
Role: {role}
Goal: {goal}
Be concise, professional, and focused only on your role."""
    }

def run_agent(agent: dict, message: str, context: str = "") -> str:
    messages = [
        {"role": "system", "content": agent["system_prompt"]}
    ]
    
    if context:
        messages.append({
            "role": "user", 
            "content": f"Context from previous agent:\n{context}\n\nYour task: {message}"
        })
    else:
        messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    return response.choices[0].message.content

# ── CREATE AGENTS ───────────────────────────────────

researcher = create_agent(
    name="ResearchAgent",
    role="AI Research Specialist",
    goal="Research topics thoroughly and extract key facts, data points, and insights"
)

writer = create_agent(
    name="WriterAgent", 
    role="Technical Content Writer",
    goal="Transform research into clear, structured, professional content"
)

critic = create_agent(
    name="CriticAgent",
    role="Quality Reviewer",
    goal="Review content for accuracy, clarity, and completeness. Give improvement suggestions"
)

manager = create_agent(
    name="ManagerAgent",
    role="Project Manager",
    goal="Coordinate agents, summarize their work, and deliver final polished answer to user"
)

# ── PIPELINE ────────────────────────────────────────

def run_pipeline(user_request: str) -> str:
    print(f"\n{'='*50}")
    print(f"📋 User Request: {user_request}")
    print(f"{'='*50}\n")
    
    # Step 1 — Research
    print("🔍 ResearchAgent working...")
    research = run_agent(
        researcher,
        f"Research this topic thoroughly: {user_request}"
    )
    print(f"📊 Research complete — {len(research.split())} words\n")
    
    # Step 2 — Write
    print("✍️  WriterAgent working...")
    written = run_agent(
        writer,
        f"Write professional content about: {user_request}",
        context=research
    )
    print(f"📝 Writing complete — {len(written.split())} words\n")
    
    # Step 3 — Review
    print("🔎 CriticAgent reviewing...")
    review = run_agent(
        critic,
        f"Review this content about: {user_request}",
        context=written
    )
    print(f"✅ Review complete\n")
    
    # Step 4 — Final answer
    print("👔 ManagerAgent finalizing...")
    final = run_agent(
        manager,
        f"Deliver the best final answer for: {user_request}",
        context=f"Research:\n{research}\n\nWritten Content:\n{written}\n\nReview:\n{review}"
    )
    
    return final

# ── MAIN ────────────────────────────────────────────

print("🤖 Multi-Agent System — Day 6")
print("==============================")
print("Agents: Researcher → Writer → Critic → Manager\n")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Your Request: ")
    
    if user_input.lower() == "quit":
        print("Multi-Agent System shutting down. See you on Day 7!")
        break
    
    result = run_pipeline(user_input)
    
    print(f"\n{'='*50}")
    print("🎯 FINAL ANSWER:")
    print(f"{'='*50}")
    print(result)
    print(f"{'='*50}\n")
