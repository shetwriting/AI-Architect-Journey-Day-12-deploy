from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Memory — stores full conversation
messages = [
    {"role": "system", "content": "You are a helpful AI Architecture tutor. Teach clearly and simply."}
]

print("AI Architect Tutor — Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye! See you on Day 3.")
        break
    
    # Add user message to memory
    messages.append({"role": "user", "content": user_input})
    
    # Send full conversation to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    reply = response.choices[0].message.content
    
    # Add AI reply to memory
    messages.append({"role": "assistant", "content": reply})
    
    print(f"\nAI: {reply}\n")




