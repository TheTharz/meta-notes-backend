from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import logging
from vectorstore import upsert_notes, query_notes
import logging.handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# add a rotating file handler
log_file = "meta_note.log"
handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=1000000, backupCount=3
)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY environment variable.")
    raise ValueError("Missing GEMINI_API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def chunk_text(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/upload/")
async def upload_notes(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)
    logger.info(f"Chunked text into {len(chunks)} parts.")
    upsert_notes(chunks)
    logger.info("Notes indexed into vector database.")
    return {"message": "Notes uploaded and indexed"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    logger.info(f"Received question: {question}")
    context_chunks = query_notes(question, top_k=3)
    context = "\n".join(context_chunks)
    logger.debug(f"Context:\n{context}")

    prompt = f"Answer the following question using only the notes below:\n\n{context}\n\nQuestion: {question}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        logger.info(f"Request to Gemini: {payload}")
        response = requests.post(GEMINI_URL, json=payload, headers={
            "Content-Type": "application/json"
        })
        response.raise_for_status()
        data = response.json()
        logger.info(f"Response from Gemini: {data}")
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        logger.info("Received answer from Gemini.")
        return {"answer": answer}

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request to Gemini failed: {e}")
        return {"error": "Failed to reach Gemini API"}

    except Exception as e:
        logger.exception("Unexpected error while handling ask_question")
        return {"error": "Failed to generate response", "raw": data if 'data' in locals() else None}
