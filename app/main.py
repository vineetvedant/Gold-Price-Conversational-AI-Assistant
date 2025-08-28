# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from app.model import get_response, CHAT_MODEL, OR_BASE_URL, is_purchase_intent

app = FastAPI(
    title="Gold Price Conversational Assistant",
    description="Chat-based assistant using Ollama + DeepSeek R1 + Chroma + RAG",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Request/Response Models --------
class QueryRequest(BaseModel):
    question: str

class ChatReply(BaseModel):
    answer: str
    next_action: str | None = None
    next_url: str | None = None

# (Kept only for reference; API 1 no longer exposes /purchase)
class PurchaseRequest(BaseModel):
    name: str = Field(..., min_length=2)
    phone: str = Field(..., min_length=8)
    amount_in_inr: float | None = Field(None, gt=0)
    grams: float | None = Field(None, gt=0)

# -------- Endpoints --------
@app.post("/chat", response_model=ChatReply)
async def chat(request: QueryRequest):
    """
    Conversational endpoint. If user consents to buy, we hand off to API 2 (purchase service).
    """
    answer = get_response(request.question)
    reply = ChatReply(answer=answer)

    if is_purchase_intent(request.question):
        reply.next_action = "purchase"
        # point to API 2 running separately
        reply.next_url = "http://127.0.0.1:8002/purchase"

    return reply

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "backend": "ollama",
        "base_url": OR_BASE_URL,
        "chat_model": CHAT_MODEL,
        "embedding_mode": "local all-MiniLM-L6-v2 (Chroma ONNX)",
        "purchase_endpoint": "http://127.0.0.1:8001/purchase",
    }
