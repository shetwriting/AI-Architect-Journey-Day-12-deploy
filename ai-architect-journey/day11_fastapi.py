from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime

load_dotenv()

# ── APP SETUP ────────────────────────────────────────
app = FastAPI(
    title="AI Architect API",
    description="Production REST API for AI Architect Tutor",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ── DATA MODELS ──────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    timestamp: str
    tokens_used: int

class RAGRequest(BaseModel):
    question: str
    documents: list[str]

class AgentRequest(BaseModel):
    task: str
    tools: list[str] = ["calculator", "datetime"]

# ── SESSION MEMORY ───────────────────────────────────
sessions = {}

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": """You are an elite AI Architect tutor.
Your student is building towards ₹1CR+ salary.
They have completed 10 days of projects.
Be concise, practical, and encouraging."""}
        ]
    return sessions[session_id]

# ── ROUTES ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>AI Architect API</title>
        <style>
            body { font-family: monospace; background: #0f0f1a; color: #00d4ff; 
                   padding: 40px; }
            h1 { font-size: 28px; margin-bottom: 10px; }
            p { color: #888; margin: 5px 0; }
            .endpoint { background: #1a1a2e; padding: 12px; margin: 8px 0; 
                        border-radius: 8px; border-left: 3px solid #00d4ff; }
            .method { color: #68d391; font-weight: bold; }
            a { color: #00d4ff; }
        </style>
    </head>
    <body>
        <h1>🤖 AI Architect API v1.0</h1>
        <p>Production REST API — Day 11</p>
        <br>
        <div class="endpoint">
            <span class="method">GET</span> /health — API health check
        </div>
        <div class="endpoint">
            <span class="method">POST</span> /chat — AI chat with memory
        </div>
        <div class="endpoint">
            <span class="method">POST</span> /rag — RAG document Q&A
        </div>
        <div class="endpoint">
            <span class="method">POST</span> /agent — AI agent with tools
        </div>
        <div class="endpoint">
            <span class="method">GET</span> /sessions — List active sessions
        </div>
        <br>
        <p>📚 Interactive docs: <a href="/docs">/docs</a></p>
        <p>📊 Alternative docs: <a href="/redoc">/redoc</a></p>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "llama-3.3-70b-versatile",
        "active_sessions": len(sessions),
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    start = time.time()
    history = get_session(request.session_id)
    history.append({"role": "user", "content": request.message})
    
    response = llm.invoke(history)
    reply = response.content
    
    history.append({"role": "assistant", "content": reply})
    
    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        timestamp=datetime.now().isoformat(),
        tokens_used=len(request.message.split()) + len(reply.split())
    )

@app.post("/rag")
async def rag_endpoint(request: RAGRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    context = "\n".join([f"- {doc}" for doc in request.documents])
    
    template = ChatPromptTemplate.from_messages([
        ("system", f"Answer based only on this context:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = template | llm
    response = chain.invoke({"question": request.question})
    
    return {
        "question": request.question,
        "answer": response.content,
        "sources_used": len(request.documents),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    available_tools = {
        "calculator": "performs math calculations",
        "datetime": "gets current date and time",
        "search": "searches knowledge base",
    }
    
    requested_tools = {
        k: v for k, v in available_tools.items() 
        if k in request.tools
    }
    
    tools_desc = "\n".join([f"- {k}: {v}" for k, v in requested_tools.items()])
    
    template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an AI Agent with these tools:
{tools_desc}
Complete the task using available tools. Be specific and actionable."""),
        ("human", "{task}")
    ])
    
    chain = template | llm
    response = chain.invoke({"task": request.task})
    
    return {
        "task": request.task,
        "result": response.content,
        "tools_available": list(requested_tools.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/sessions")
async def list_sessions():
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys()),
        "total_messages": sum(len(s) for s in sessions.values())
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted"}

