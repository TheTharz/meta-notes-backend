# api/ask.py
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import os
import requests
from vectorstore import query_notes

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    context_chunks = query_notes(question, top_k=3)
    context = "\n".join(context_chunks)

    prompt = f"Answer the following question using only the notes below:\n\n{context}\n\nQuestion: {question}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(GEMINI_URL, json=payload, headers={
        "Content-Type": "application/json"
    })

    data = response.json()
    try:
        return JSONResponse(content={"answer": data["candidates"][0]["content"]["parts"][0]["text"]})
    except:
        return JSONResponse(status_code=500, content={"error": "Failed to generate response", "raw": data})
