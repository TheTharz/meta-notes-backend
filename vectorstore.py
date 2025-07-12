import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from embedding import embed_text

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_REGION = os.getenv("PINECONE_REGION")  # e.g. "us-east-1"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")  # Optional, default to "aws"

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # or 768 depending on your model
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

# Connect to index
index = pc.Index(PINECONE_INDEX)

def upsert_notes(note_chunks: list[str]):
    embeddings = embed_text(note_chunks)
    to_upsert = [(f"note-{i}", embeddings[i], {"text": note_chunks[i]}) for i in range(len(note_chunks))]
    index.upsert(vectors=to_upsert)

def query_notes(query: str, top_k: int = 3) -> list[str]:
    vector = embed_text([query])[0]
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
