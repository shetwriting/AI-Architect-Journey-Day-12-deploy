from groq import Groq
from dotenv import load_dotenv
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = [
    {"role": "system", "content": """You are a personal AI Architect tutor.
Your student is on a journey to become an AI Architect and earn ₹1CR+ salary.
They have completed:
- Day 1: Groq API + First Script
- Day 2: Memory Chatbot
- Day 3: Persistent AI Tutor
- Day 4: RAG System
- Day 5: AI Agent with Tools
- Day 6: Multi-Agent Pipeline
- Day 7: Full Stack Web App (current)
Be encouraging, teach clearly, and relate everything to real AI Architect skills."""}
]

def chat_with_ai(user_message):
    conversation_history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Architect Tutor</title>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #080810;
            --surface: #0e0e1c;
            --surface2: #13131f;
            --border: rgba(255,255,255,0.06);
            --border-bright: rgba(99,179,237,0.3);
            --accent: #63b3ed;
            --accent2: #9f7aea;
            --accent3: #68d391;
            --text: #e8e8f0;
            --text-dim: #6b6b8a;
            --text-muted: #3a3a55;
            --user-bg: rgba(99,179,237,0.08);
            --ai-bg: rgba(255,255,255,0.03);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Mono', monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        /* Ambient background */
        body::before {
            content: '';
            position: fixed;
            top: -40%;
            left: -20%;
            width: 60%;
            height: 80%;
            background: radial-gradient(ellipse, rgba(99,179,237,0.04) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }
        body::after {
            content: '';
            position: fixed;
            bottom: -30%;
            right: -10%;
            width: 50%;
            height: 70%;
            background: radial-gradient(ellipse, rgba(159,122,234,0.04) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* Header */
        header {
            position: relative;
            z-index: 10;
            padding: 18px 28px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(8,8,16,0.8);
            backdrop-filter: blur(20px);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .logo-text {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 16px;
            letter-spacing: -0.3px;
            color: var(--text);
        }

        .logo-text span {
            color: var(--accent);
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent3);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }

        .status-text {
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.5px;
        }

        /* Progress strip */
        .progress-strip {
            position: relative;
            z-index: 10;
            padding: 10px 28px;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 6px;
            overflow-x: auto;
            scrollbar-width: none;
            background: rgba(8,8,16,0.6);
        }

        .progress-strip::-webkit-scrollbar { display: none; }

        .day-chip {
            flex-shrink: 0;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 10px;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.3px;
            border: 1px solid var(--border);
            color: var(--text-dim);
            background: transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .day-chip.done {
            border-color: rgba(104,211,145,0.25);
            color: var(--accent3);
            background: rgba(104,211,145,0.05);
        }

        .day-chip.current {
            border-color: var(--accent);
            color: var(--accent);
            background: rgba(99,179,237,0.08);
        }

        /* Main chat area */
        main {
            flex: 1;
            overflow-y: auto;
            padding: 24px 28px;
            position: relative;
            z-index: 5;
            scroll-behavior: smooth;
        }

        main::-webkit-scrollbar { width: 4px; }
        main::-webkit-scrollbar-track { background: transparent; }
        main::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 2px; }

        .messages-container {
            max-width: 760px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        /* Welcome message */
        .welcome-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }

        .welcome-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
        }

        .welcome-title {
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            font-size: 18px;
            margin-bottom: 8px;
            color: var(--text);
        }

        .welcome-title span { color: var(--accent); }

        .welcome-body {
            font-size: 12px;
            color: var(--text-dim);
            line-height: 1.7;
        }

        .week-stats {
            display: flex;
            gap: 16px;
            margin-top: 16px;
        }

        .stat {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .stat-value {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 22px;
            color: var(--accent);
        }

        .stat-label {
            font-size: 10px;
            color: var(--text-dim);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Message bubbles */
        .message-row {
            display: flex;
            gap: 12px;
            padding: 6px 0;
            animation: fadeSlideIn 0.3s ease forwards;
            opacity: 0;
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-row.user {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            margin-top: 2px;
        }

        .avatar.ai {
            background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(159,122,234,0.2));
            border: 1px solid rgba(99,179,237,0.2);
        }

        .avatar.user {
            background: linear-gradient(135deg, rgba(104,211,145,0.15), rgba(99,179,237,0.15));
            border: 1px solid rgba(104,211,145,0.2);
        }

        .bubble {
            max-width: 78%;
            padding: 12px 16px;
            border-radius: 14px;
            font-size: 13px;
            line-height: 1.75;
            position: relative;
        }

        .message-row.ai .bubble {
            background: var(--ai-bg);
            border: 1px solid var(--border);
            border-top-left-radius: 4px;
            color: var(--text);
        }

        .message-row.user .bubble {
            background: var(--user-bg);
            border: 1px solid var(--border-bright);
            border-top-right-radius: 4px;
            color: var(--text);
            text-align: right;
        }

        .bubble b { color: var(--accent); font-weight: 600; }

        .bubble-meta {
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 6px;
            letter-spacing: 0.3px;
        }

        .message-row.user .bubble-meta { text-align: right; }

        /* Thinking animation */
        .thinking-dots {
            display: flex;
            gap: 4px;
            align-items: center;
            padding: 4px 0;
        }

        .thinking-dots span {
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--accent);
            opacity: 0.4;
            animation: dotBounce 1.2s ease-in-out infinite;
        }

        .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes dotBounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
            40% { transform: translateY(-4px); opacity: 1; }
        }

        /* Input area */
        footer {
            position: relative;
            z-index: 10;
            padding: 16px 28px 20px;
            border-top: 1px solid var(--border);
            background: rgba(8,8,16,0.9);
            backdrop-filter: blur(20px);
        }

        .input-wrapper {
            max-width: 760px;
            margin: 0 auto;
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        .input-box {
            flex: 1;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 13px 16px;
            color: var(--text);
            font-family: 'DM Mono', monospace;
            font-size: 13px;
            outline: none;
            resize: none;
            min-height: 46px;
            max-height: 120px;
            transition: border-color 0.2s;
            line-height: 1.5;
        }

        .input-box::placeholder { color: var(--text-muted); }

        .input-box:focus {
            border-color: rgba(99,179,237,0.4);
            background: var(--surface2);
        }

        .send-btn {
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            flex-shrink: 0;
        }

        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(99,179,237,0.3);
        }

        .send-btn:active { transform: translateY(0); }

        .send-btn svg {
            width: 18px;
            height: 18px;
            fill: white;
        }

        .input-hint {
            text-align: center;
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 8px;
            letter-spacing: 0.3px;
        }

        /* Suggestion chips */
        .suggestions {
            max-width: 760px;
            margin: 0 auto 12px;
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }

        .suggestion-chip {
            padding: 5px 11px;
            border-radius: 8px;
            font-size: 11px;
            font-family: 'DM Mono', monospace;
            border: 1px solid var(--border);
            color: var(--text-dim);
            background: transparent;
            cursor: pointer;
            transition: all 0.15s;
        }

        .suggestion-chip:hover {
            border-color: var(--border-bright);
            color: var(--accent);
            background: rgba(99,179,237,0.05);
        }

        /* Scrollbar for main */
        * { scrollbar-width: thin; scrollbar-color: var(--text-muted) transparent; }
    </style>
</head>
<body>

<header>
    <div class="logo">
        <div class="logo-icon">🤖</div>
        <div class="logo-text">AI <span>Architect</span> Tutor</div>
    </div>
    <div class="header-right">
        <div class="status-dot"></div>
        <span class="status-text">GROQ · llama-3.3-70b</span>
    </div>
</header>

<div class="progress-strip">
    <div class="day-chip done">✓ D1 · API</div>
    <div class="day-chip done">✓ D2 · Memory</div>
    <div class="day-chip done">✓ D3 · Persist</div>
    <div class="day-chip done">✓ D4 · RAG</div>
    <div class="day-chip done">✓ D5 · Agent</div>
    <div class="day-chip done">✓ D6 · Multi-Agent</div>
    <div class="day-chip current">● D7 · Web App</div>
    <div class="day-chip">D8 · LangChain</div>
    <div class="day-chip">D9 · VectorDB</div>
    <div class="day-chip">D10 · Fine-tune</div>
</div>

<main id="main">
    <div class="messages-container" id="chat-box">

        <div class="welcome-card">
            <div class="welcome-title">Week 1 <span>Complete</span> 🏆</div>
            <div class="welcome-body">
                You built 7 real AI systems in 7 days. Most developers never build any of this.<br>
                Ask me anything about your journey, next steps, or AI concepts.
            </div>
            <div class="week-stats">
                <div class="stat">
                    <div class="stat-value">7</div>
                    <div class="stat-label">Projects</div>
                </div>
                <div class="stat">
                    <div class="stat-value">7</div>
                    <div class="stat-label">Days</div>
                </div>
                <div class="stat">
                    <div class="stat-value">₹1CR</div>
                    <div class="stat-label">Goal</div>
                </div>
            </div>
        </div>

    </div>
</main>

<footer>
    <div class="suggestions" id="suggestions">
        <button class="suggestion-chip" onclick="usesuggestion(this)">What should I learn in Week 2?</button>
        <button class="suggestion-chip" onclick="usesuggestion(this)">How close am I to ₹1CR salary?</button>
        <button class="suggestion-chip" onclick="usesuggestion(this)">Explain RAG in simple terms</button>
        <button class="suggestion-chip" onclick="usesuggestion(this)">What have I built this week?</button>
    </div>

    <div class="input-wrapper">
        <textarea
            id="user-input"
            class="input-box"
            placeholder="Ask your AI tutor anything..."
            rows="1"
            onkeydown="handleKey(event)"
            oninput="autoResize(this)"
        ></textarea>
        <button class="send-btn" onclick="sendMessage()">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
        </button>
    </div>
    <div class="input-hint">Enter to send · Shift+Enter for new line</div>
</footer>

<script>
    function getTime() {
        return new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }

    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }

    function handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    function usesuggestion(btn) {
        document.getElementById('user-input').value = btn.textContent;
        document.getElementById('suggestions').style.display = 'none';
        sendMessage();
    }

    function appendMessage(role, content) {
        const box = document.getElementById('chat-box');
        const time = getTime();
        const isAI = role === 'ai';

        const formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
            .replace(/\n/g, '<br>');

        const row = document.createElement('div');
        row.className = `message-row ${role}`;
        row.innerHTML = `
            <div class="avatar ${role}">${isAI ? '🤖' : '👤'}</div>
            <div class="bubble">
                ${formatted}
                <div class="bubble-meta">${time}</div>
            </div>
        `;
        box.appendChild(row);
        scrollToBottom();
        return row;
    }

    function showThinking() {
        const box = document.getElementById('chat-box');
        const row = document.createElement('div');
        row.className = 'message-row ai';
        row.id = 'thinking-row';
        row.innerHTML = `
            <div class="avatar ai">🤖</div>
            <div class="bubble">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        box.appendChild(row);
        scrollToBottom();
    }

    function removeThinking() {
        const el = document.getElementById('thinking-row');
        if (el) el.remove();
    }

    function scrollToBottom() {
        const main = document.getElementById('main');
        main.scrollTop = main.scrollHeight;
    }

    async function sendMessage() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();
        if (!message) return;

        // Hide suggestions after first message
        document.getElementById('suggestions').style.display = 'none';

        appendMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        showThinking();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message})
            });
            const data = await response.json();
            removeThinking();
            appendMessage('ai', data.reply);
        } catch (err) {
            removeThinking();
            appendMessage('ai', 'Connection error. Make sure the server is running.');
        }
    }
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML.encode('utf-8'))

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        reply = chat_with_ai(body['message'])
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"reply": reply}).encode())

print("🚀 AI Architect Tutor Web App")
print("==============================")
print("✅ Server starting...")
print("🌐 Open your browser: http://localhost:8000")
print("\nPress Ctrl+C to stop\n")

HTTPServer(('', 8000), Handler).serve_forever()
