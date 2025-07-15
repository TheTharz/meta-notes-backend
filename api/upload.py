# api/upload.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from vectorstore import upsert_notes
import io
import os
import fitz  # PyMuPDF

app = FastAPI()

def chunk_text(text, chunk_size=400):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text_from_pdf(file: UploadFile) -> str:
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

@app.post("/api/upload")
async def upload_notes(file: UploadFile = File(...)):
    filename = file.filename.lower()
    
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".txt"):
        content = await file.read()
        text = content.decode("utf-8")
    else:
        return JSONResponse(content={"error": "Only .txt and .pdf files are supported"}, status_code=400)

    chunks = chunk_text(text)
    upsert_notes(chunks)
    return JSONResponse(content={"message": "Notes uploaded and indexed"})
