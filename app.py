import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import generate_response

app = FastAPI(title="Aaple Sarkar Chatbot API")

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(None)):
    expected_key = os.getenv("CHATBOT_API_KEY")

    if not expected_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aaplesarkar2.maharashtra.gov.in"],
    allow_methods=["*"],
    allow_headers=["*", "x-api-key"],
)

# ================= SCHEMAS =================
class ChatRequest(BaseModel):
    query: str
    language: str = "en"

# ================= ROUTES =================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)
    answer = generate_response(req.query, req.language)
    return {"answer": answer}