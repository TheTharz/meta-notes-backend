# api/upload.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from vectorstore import upsert_notes
import io

app = FastAPI()

def chunk_text(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/api/upload")
async def upload_notes(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)
    upsert_notes(chunks)
    return JSONResponse(content={"message": "Notes uploaded and indexed"})
