from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
from vectorstore import upsert_notes, query_notes

app = FastAPI()

# Enable CORS for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple chunker
def chunk_text(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/upload/")
async def upload_notes(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)
    upsert_notes(chunks)
    return {"message": "Notes uploaded and indexed"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    context_chunks = query_notes(question, top_k=3)
    context = "\n".join(context_chunks)

    prompt = f"Answer the following question using only the notes below:\n\n{context}\n\nQuestion: {question}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    from config import GEMINI_URL

    response = requests.post(GEMINI_URL, json=payload, headers={
        "Content-Type": "application/json"
    })

    data = response.json()
    try:
        return {"answer": data["candidates"][0]["content"]["parts"][0]["text"]}
    except:
        return {"error": "Failed to generate response", "raw": data}
