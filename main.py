"""
ΛXIOM — Online AI Assistant Backend
FastAPI + Supabase + Gemini + Groq + OpenRouter
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import httpx, json, uuid, os, hashlib
from datetime import datetime, timedelta
import jwt

# ── ENV VARS — set all in Render dashboard ──
SUPABASE_URL   = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # aistudio.google.com/app/apikey
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")     # console.groq.com
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")   # openrouter.ai/keys
SECRET_KEY     = os.getenv("SECRET_KEY", "change-me-please")

app = FastAPI(title="ΛXIOM API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer()

# ─────────────────────────────────────────
#  SUPABASE
# ─────────────────────────────────────────

def sb_headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json", "Prefer": "return=representation"}

async def sb_get(table, query=""):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{SUPABASE_URL}/rest/v1/{table}{query}", headers=sb_headers())
        return r.json()

async def sb_post(table, data):
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=sb_headers(), json=data)
        return r.json()

async def sb_patch(table, query, data):
    async with httpx.AsyncClient() as c:
        await c.patch(f"{SUPABASE_URL}/rest/v1/{table}{query}", headers=sb_headers(), json=data)

async def sb_delete(table, query):
    async with httpx.AsyncClient() as c:
        await c.delete(f"{SUPABASE_URL}/rest/v1/{table}{query}", headers=sb_headers())

# ─────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────

ALL_MODELS = {
    # Google Gemini — most generous free tier, huge context
    "gemini-2.0-flash": {"id": "gemini-2.0-flash-exp",          "name": "Gemini 2.0 Flash",  "provider": "gemini",     "desc": "Latest Google model, 1M context", "badge": "NEW"},
    "gemini-1.5-flash": {"id": "gemini-1.5-flash",              "name": "Gemini 1.5 Flash",  "provider": "gemini",     "desc": "Fast, massive 1M token context",  "badge": ""},
    "gemini-1.5-pro":   {"id": "gemini-1.5-pro",                "name": "Gemini 1.5 Pro",    "provider": "gemini",     "desc": "Most powerful Gemini model",      "badge": "PRO"},
    "gemma-2-9b":       {"id": "gemma-2-9b-it",                 "name": "Gemma 2 9B",        "provider": "gemini",     "desc": "Google open model, lightweight",  "badge": ""},
    # Groq — ultra-fast inference
    "llama3-70b":       {"id": "llama3-70b-8192",               "name": "LLaMA 3 70B",       "provider": "groq",       "desc": "Powerful open model, fast",       "badge": "FAST"},
    "llama3-8b":        {"id": "llama3-8b-8192",                "name": "LLaMA 3 8B",        "provider": "groq",       "desc": "Blazing fast responses",          "badge": "FAST"},
    "mixtral-8x7b":     {"id": "mixtral-8x7b-32768",            "name": "Mixtral 8x7B",      "provider": "groq",       "desc": "32K context, mixture of experts", "badge": ""},
    # OpenRouter — extra model variety
    "mistral-7b":       {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B",        "provider": "openrouter", "desc": "Balanced, multilingual support",  "badge": ""},
    "qwen2-72b":        {"id": "qwen/qwen-2-72b-instruct",      "name": "Qwen 2 72B",        "provider": "openrouter", "desc": "Best multilingual + coding",      "badge": ""},
    "deepseek-v2":      {"id": "deepseek/deepseek-chat",        "name": "DeepSeek V2",       "provider": "openrouter", "desc": "Strong coding & reasoning",       "badge": ""},
}

DEFAULT_MODEL = "gemini-2.0-flash"

TASK_PRESETS = {
    "general":      "You are a friendly, helpful assistant. Be conversational, clear, and concise.",
    "coding":       "You are an expert software engineer. Help with code, debugging, and architecture. Always format code properly.",
    "writing":      "You are a professional writer and editor. Help with essays, emails, creative writing, and grammar.",
    "study":        "You are a patient tutor. Explain complex topics clearly with examples. Adapt to the user's level.",
    "multilingual": "You are a multilingual assistant. Always detect the user's language and respond in the same language.",
    "research":     "You are an analytical research assistant. Give thorough, structured, nuanced responses. Think step by step.",
}

# ─────────────────────────────────────────
#  PROVIDER STREAMING
# ─────────────────────────────────────────

async def stream_gemini(model_id, messages, system):
    """Stream from Google AI Studio (Gemini native API)."""
    # Convert to Gemini format
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in messages]
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}")
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.7}
    }
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    token = json.loads(raw)["candidates"][0]["content"]["parts"][0].get("text", "")
                    if token:
                        yield token
                except Exception:
                    continue

async def stream_openai_compat(url, headers, model_id, messages):
    """Stream from any OpenAI-compatible endpoint (Groq, OpenRouter)."""
    payload = {"model": model_id, "messages": messages, "stream": True, "max_tokens": 3000}
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    token = json.loads(raw)["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue

# ─────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def make_token(uid, username, role):
    return jwt.encode({"sub": uid, "username": username, "role": role,
                       "exp": datetime.utcnow() + timedelta(days=30)}, SECRET_KEY, algorithm="HS256")

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

# ─────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────

class LoginReq(BaseModel):
    username: str
    password: str

class ChatReq(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    preset: str = "general"
    chat_id: Optional[str] = None

class NewChatReq(BaseModel):
    title: str = "New Chat"
    model: str = DEFAULT_MODEL
    preset: str = "general"

# ─────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: LoginReq):
    if await sb_get("users", f"?username=eq.{req.username}&select=id"):
        raise HTTPException(400, "Username already taken")
    uid = str(uuid.uuid4())
    all_users = await sb_get("users", "?select=id")
    role = "admin" if not all_users else "user"
    await sb_post("users", {"id": uid, "username": req.username,
                             "password_hash": hash_pw(req.password), "role": role})
    return {"token": make_token(uid, req.username, role), "username": req.username, "role": role}

@app.post("/api/auth/login")
async def login(req: LoginReq):
    users = await sb_get("users", f"?username=eq.{req.username}&password_hash=eq.{hash_pw(req.password)}&select=*")
    if not users:
        raise HTTPException(401, "Invalid credentials")
    u = users[0]
    return {"token": make_token(u["id"], u["username"], u["role"]), "username": u["username"], "role": u["role"]}

# ─────────────────────────────────────────
#  MODEL / PRESET ROUTES
# ─────────────────────────────────────────

@app.get("/api/models")
async def list_models(user=Depends(verify_token)):
    return {"models": [{"key": k, **v} for k, v in ALL_MODELS.items()], "default": DEFAULT_MODEL}

@app.get("/api/presets")
async def list_presets(user=Depends(verify_token)):
    return {"presets": list(TASK_PRESETS.keys())}

# ─────────────────────────────────────────
#  CHAT ROUTES
# ─────────────────────────────────────────

@app.get("/api/chats")
async def get_chats(user=Depends(verify_token)):
    return {"chats": await sb_get("chats", f"?user_id=eq.{user['sub']}&order=updated_at.desc&select=*")}

@app.post("/api/chats")
async def create_chat(req: NewChatReq, user=Depends(verify_token)):
    cid = str(uuid.uuid4())
    await sb_post("chats", {"id": cid, "user_id": user["sub"], "title": req.title, "model": req.model, "preset": req.preset})
    return {"chat_id": cid}

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user=Depends(verify_token)):
    chats = await sb_get("chats", f"?id=eq.{chat_id}&user_id=eq.{user['sub']}&select=*")
    if not chats:
        raise HTTPException(404, "Chat not found")
    return {"chat": chats[0], "messages": await sb_get("messages", f"?chat_id=eq.{chat_id}&order=created_at.asc&select=*")}

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user=Depends(verify_token)):
    await sb_delete("messages", f"?chat_id=eq.{chat_id}")
    await sb_delete("chats",    f"?id=eq.{chat_id}&user_id=eq.{user['sub']}")
    return {"ok": True}

@app.post("/api/chats/{chat_id}/share")
async def share_chat(chat_id: str, user=Depends(verify_token)):
    token = str(uuid.uuid4())
    await sb_patch("chats", f"?id=eq.{chat_id}&user_id=eq.{user['sub']}", {"share_token": token})
    return {"share_token": token}

@app.get("/api/shared/{share_token}")
async def get_shared(share_token: str):
    chats = await sb_get("chats", f"?share_token=eq.{share_token}&select=*")
    if not chats:
        raise HTTPException(404, "Not found")
    return {"chat": chats[0], "messages": await sb_get("messages", f"?chat_id=eq.{chats[0]['id']}&order=created_at.asc&select=*")}

# ─────────────────────────────────────────
#  STREAMING CHAT
# ─────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq, user=Depends(verify_token)):
    model_info    = ALL_MODELS.get(req.model, ALL_MODELS[DEFAULT_MODEL])
    system_prompt = TASK_PRESETS.get(req.preset, TASK_PRESETS["general"])
    provider      = model_info["provider"]

    # Get or create chat
    chat_id = req.chat_id
    if not chat_id:
        chat_id = str(uuid.uuid4())
        title = req.message[:45] + ("..." if len(req.message) > 45 else "")
        await sb_post("chats", {"id": chat_id, "user_id": user["sub"],
                                "title": title, "model": req.model, "preset": req.preset})

    await sb_post("messages", {"id": str(uuid.uuid4()), "chat_id": chat_id, "role": "user", "content": req.message})
    await sb_patch("chats", f"?id=eq.{chat_id}", {"updated_at": datetime.utcnow().isoformat()})

    history = await sb_get("messages", f"?chat_id=eq.{chat_id}&order=created_at.asc&select=role,content")
    oai_msgs = [{"role": "system", "content": system_prompt}] + \
               [{"role": m["role"], "content": m["content"]} for m in history]
    gem_msgs = [{"role": m["role"], "content": m["content"]} for m in history]

    async def generate():
        full = ""
        yield f"data: {json.dumps({'type':'start','chat_id':chat_id,'provider':provider})}\n\n"
        try:
            if provider == "gemini":
                async for tok in stream_gemini(model_info["id"], gem_msgs, system_prompt):
                    full += tok
                    yield f"data: {json.dumps({'type':'token','token':tok})}\n\n"

            elif provider == "groq":
                hdrs = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                async for tok in stream_openai_compat("https://api.groq.com/openai/v1/chat/completions", hdrs, model_info["id"], oai_msgs):
                    full += tok
                    yield f"data: {json.dumps({'type':'token','token':tok})}\n\n"

            elif provider == "openrouter":
                hdrs = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json",
                        "HTTP-Referer": "https://axiom-ai.vercel.app", "X-Title": "ΛXIOM"}
                async for tok in stream_openai_compat("https://openrouter.ai/api/v1/chat/completions", hdrs, model_info["id"], oai_msgs):
                    full += tok
                    yield f"data: {json.dumps({'type':'token','token':tok})}\n\n"

            await sb_post("messages", {"id": str(uuid.uuid4()), "chat_id": chat_id, "role": "assistant", "content": full})
            await sb_post("stats",    {"id": str(uuid.uuid4()), "user_id": user["sub"],
                                       "model": req.model, "preset": req.preset, "provider": provider})
            yield f"data: {json.dumps({'type':'done','chat_id':chat_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ─────────────────────────────────────────
#  ADMIN
# ─────────────────────────────────────────

@app.get("/api/admin/stats")
async def admin_stats(user=Depends(verify_token)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    users    = await sb_get("users",    "?select=id,username,role,created_at&order=created_at.desc")
    chats    = await sb_get("chats",    "?select=id")
    messages = await sb_get("messages", "?select=id")
    stats    = await sb_get("stats",    "?select=model,preset,provider")

    model_counts = {}; preset_counts = {}; provider_counts = {}
    for s in stats:
        model_counts[s["model"]]              = model_counts.get(s["model"], 0) + 1
        preset_counts[s["preset"]]            = preset_counts.get(s["preset"], 0) + 1
        provider_counts[s.get("provider","?")] = provider_counts.get(s.get("provider","?"), 0) + 1

    return {
        "total_users":    len(users),
        "total_chats":    len(chats),
        "total_messages": len(messages),
        "model_usage":    sorted([{"model": k,    "count": v} for k,v in model_counts.items()],    key=lambda x: -x["count"]),
        "preset_usage":   sorted([{"preset": k,   "count": v} for k,v in preset_counts.items()],   key=lambda x: -x["count"]),
        "provider_usage": sorted([{"provider": k, "count": v} for k,v in provider_counts.items()], key=lambda x: -x["count"]),
        "recent_users":   users[:8],
    }

@app.get("/api/health")
async def health():
    return {"status": "ok", "providers": {
        "gemini": bool(GEMINI_API_KEY),
        "groq": bool(GROQ_API_KEY),
        "openrouter": bool(OPENROUTER_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
    }, "time": datetime.utcnow().isoformat()}
