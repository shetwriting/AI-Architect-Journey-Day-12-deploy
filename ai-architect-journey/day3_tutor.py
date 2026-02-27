from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MEMORY_FILE = "tutor_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return [
        {"role": "system", "content": """You are a personal AI Architect tutor. 
        Your student is learning to become an AI Architect to earn ₹1CR+ salary.
        Remember their progress, encourage them, and teach step by step.
        Always relate concepts to real AI Architect job skills."""}
    ]

def save_memory(messages):
    with open(MEMORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)

def chat(messages, user_input):
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    
    return reply, messages

# Main program
print("🤖 Personal AI Architect Tutor")
print("================================")

messages = load_memory()

if len(messages) > 1:
    print(f"✅ Loaded {len(messages)-1} previous messages — continuing your journey!\n")
else:
    print("🆕 Starting fresh session!\n")

print("Type 'quit' to save and exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        save_memory(messages)
        print("\n💾 Progress saved! See you tomorrow!")
        break
    
    reply, messages = chat(messages, user_input)
    print(f"\nAI Tutor: {reply}\n")
   
