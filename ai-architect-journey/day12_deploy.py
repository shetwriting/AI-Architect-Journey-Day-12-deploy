from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import time
from datetime import datetime

load_dotenv()

# ── APP ──────────────────────────────────────────────
app = FastAPI(
    title="AI Architect API",
    description="Cloud-deployed AI API — Day 12",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── LLM ─────────────────────────────────────────────
GROQ_KEY = os.getenv("GROQ_API_KEY")

def get_llm():
    return ChatGroq(
        api_key=GROQ_KEY,
        model_name="llama-3.3-70b-versatile"
    )

# ── MODELS ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class RAGRequest(BaseModel):
    question: str
    documents: list[str]

# ── MEMORY ──────────────────────────────────────────
sessions = {}

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": """You are an elite AI Architect tutor.
Your student has completed 12 days of building AI projects:
Day 1: Groq API Script | Day 2: Memory Chatbot | Day 3: Persistent Tutor
Day 4: RAG System | Day 5: AI Agent | Day 6: Multi-Agent Pipeline  
Day 7: Web App | Day 8: LangChain | Day 9: Vector DB | Day 10: Fine-tuning
Day 11: FastAPI Backend | Day 12: Cloud Deployment (current)
Goal: AI Architect role earning 1CR+ salary.
Be concise, practical, and encouraging."""}
        ]
    return sessions[session_id]

# ── ROUTES ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Architect API</title>
        <meta charset="UTF-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Courier New', monospace; 
                background: #0f0f1a; 
                color: #00d4ff;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { max-width: 600px; padding: 40px; }
            h1 { font-size: 28px; margin-bottom: 8px; }
            .subtitle { color: #888; margin-bottom: 32px; font-size: 14px; }
            .badge { 
                display: inline-block;
                background: #00d4ff22; 
                border: 1px solid #00d4ff44;
                padding: 4px 12px; border-radius: 20px; 
                font-size: 12px; margin-bottom: 24px;
            }
            .endpoint { 
                background: #1a1a2e; padding: 14px 18px; 
                margin: 8px 0; border-radius: 8px; 
                border-left: 3px solid #00d4ff;
                display: flex; gap: 12px; align-items: center;
            }
            .method { color: #68d391; font-weight: bold; min-width: 50px; }
            .path { color: #fff; }
            .desc { color: #888; font-size: 13px; margin-left: auto; }
            .links { margin-top: 24px; }
            .links a { 
                color: #00d4ff; text-decoration: none;
                margin-right: 20px; font-size: 14px;
            }
            .links a:hover { text-decoration: underline; }
            .status { 
                margin-top: 24px; padding: 12px 18px;
                background: #00d4ff11; border-radius: 8px;
                font-size: 13px; color: #68d391;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Architect API</h1>
            <p class="subtitle">Production REST API — Deployed Day 12</p>
            <div class="badge">✅ Live on Cloud</div>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/health</span>
                <span class="desc">Status check</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/chat</span>
                <span class="desc">AI chat with memory</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/rag</span>
                <span class="desc">Document Q&A</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/sessions</span>
                <span class="desc">Active sessions</span>
            </div>
            
            <div class="links">
                <a href="/docs">📚 Swagger Docs</a>
                <a href="/health">💚 Health Check</a>
                <a href="/redoc">📖 ReDoc</a>
            </div>
            
            <div class="status">
                🚀 API running — call from anywhere in the world
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "AI Architect API is live!",
        "timestamp": datetime.now().isoformat(),
        "model": "llama-3.3-70b-versatile",
        "active_sessions": len(sessions),
        "version": "2.0.0",
        "deployed": "cloud"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    history = get_session(request.session_id)
    history.append({"role": "user", "content": request.message})
    
    llm = get_llm()
    response = llm.invoke(history)
    reply = response.content
    
    history.append({"role": "assistant", "content": reply})
    
    return {
        "reply": reply,
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/rag")
async def rag_endpoint(request: RAGRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    context = "\n".join([f"- {doc}" for doc in request.documents])
    
    template = ChatPromptTemplate.from_messages([
        ("system", f"Answer based only on this context:\n{context}"),
        ("human", "{question}")
    ])
    
    llm = get_llm()
    chain = template | llm
    response = chain.invoke({"question": request.question})
    
    return {
        "question": request.question,
        "answer": response.content,
        "sources_used": len(request.documents),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/sessions")
async def list_sessions():
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }