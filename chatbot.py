import os
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import re

# ================= ENV =================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================= CONSTANTS =================
SERVICES_INDEX_NAME = "aaple-sarkar-services"
BASICS_INDEX_NAME = "aaplesarkarbasics"

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

# ================= INIT =================
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
embedder = SentenceTransformer("all-MiniLM-L6-v2")

services_index = pc.Index(SERVICES_INDEX_NAME) if pc else None
basics_index = pc.Index(BASICS_INDEX_NAME) if pc else None

# ================= LOAD DATA =================
EXCEL_URL = "https://huggingface.co/datasets/PrathameshRaut/aaple-sarkar-data/resolve/main/Untitled%20spreadsheet-2.xlsx"
LOCAL_EXCEL_PATH = os.path.join(DATA_DIR, "services.xlsx")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(LOCAL_EXCEL_PATH):
    r = requests.get(EXCEL_URL)
    r.raise_for_status()
    with open(LOCAL_EXCEL_PATH, "wb") as f:
        f.write(r.content)

excel_df = pd.read_excel(
    LOCAL_EXCEL_PATH,
    sheet_name="Copy of Notified services (1212"
)

# ================= SEARCH =================
def is_small_talk(text: str) -> bool:
    text = text.strip().lower()
    return bool(re.fullmatch(
        r"(hi|hello|hey|hii|hai|namaste|thanks|thank you|ok|okay|yes|no|namaskar|good morning|good afternoon|good evening|good night|how are you?|how are you|how are you.|who are you?|introduce yourself|introduce)",
        text
    ))


def search_services(query, min_score=0.75):
    if not services_index:
        return None

    emb = embedder.encode([query])[0].tolist()
    res = services_index.query(
        vector=emb,
        top_k=1,
        include_metadata=True
    )

    if not res.matches:
        return None

    match = res.matches[0]
    if match.score < min_score:
        return None

    return match.metadata["text"]


def search_basics(query):
    if not basics_index:
        return ""

    emb = embedder.encode([query])[0].tolist()
    res = basics_index.query(
        vector=emb,
        top_k=5,
        include_metadata=True
    )
    return "\n\n".join(m.metadata["text"] for m in res.matches)


def get_service_info(dept, service):
    mask = (
        (excel_df["Department"].str.lower() == dept.lower()) &
        (excel_df["Service Name"]
         .str.lower()
         .str.contains(service.lower(), na=False))
    )
    row = excel_df[mask]
    if row.empty:
        return None
    return {
        k: v for k, v in row.iloc[0].to_dict().items()
        if pd.notna(v) and str(v).strip()
    }

# ================= RESPONSE =================
def generate_response(user_query, language="en"):

    # ================= GUARD =================
    if not client or not pc:
        return "<p>❌ Server misconfiguration: API keys are missing. Please contact the administrator.</p>"

    # ================= SMALL TALK SHORT-CIRCUIT =================
    if is_small_talk(user_query):
        prompt = f"""
You are Aaple Sarkar Services Chatbot (Maharashtra Government) named "Aapla Sahayak".
Rules:
- Respond politely, briefly, and naturally using some 1-2 emojis, also while greeting say Namaskar.
- Respond with RAW, valid, render-ready HTML only.
- Do not use markdown.
- Do not add explanations outside HTML.
User Query:
{user_query}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content

    # ================= RAG PIPELINE =================
    basics_context = search_basics(user_query)
    service_match = search_services(user_query)

    context_parts = []

    if basics_context:
        context_parts.append(
            f"General Aaple Sarkar Information:\n{basics_context}"
        )

    if service_match:
        try:
            dept_part, service_part = service_match.split(" - Service: ")
            dept = dept_part.replace("Department:", "").strip()
            service = service_part.strip()

            info = get_service_info(dept, service)

            context_parts.append(
                f"Detected Service:\nDepartment: {dept}\nService: {service}"
            )

            if info:
                context_parts.append(
                    "Service Details:\n" +
                    "\n".join(
                        f"{k}: {v}" for k, v in list(info.items())[:20]
                    )
                )
        except Exception:
            pass

    context = "\n\n".join(context_parts) or "No relevant data found."

    # ================= MAIN PROMPT =================
    prompt = f"""
You are Aaple Sarkar Services Chatbot (Maharashtra Government).
Rules:
- Respond with RAW, valid, render-ready HTML only; do not use markdown, do not wrap the response in ``` or ```html, do not add explanations/comments/text outside HTML, do not add leading or trailing whitespace, and ensure the response starts directly with <html> or the first HTML tag.
- If the user message is a generic, short, or social message (such as greetings, acknowledgements, or fillers), you must respond appropriately with a polite, natural, and context-independent reply.
User Query:
{user_query}
Available Knowledge:
{context}
Provide complete guidance including:
- Eligibility
- Documents required
- Fees
- Timeline
- Officers involved
- Online application steps
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.6,
        max_tokens=800,
    )

    return response.choices[0].message.content