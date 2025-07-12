import pinecone
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX
from embedding import embed_text

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

def upsert_notes(note_chunks: list[str]):
    embeddings = embed_text(note_chunks)
    to_upsert = [(f"note-{i}", embeddings[i], {"text": note_chunks[i]}) for i in range(len(note_chunks))]
    index.upsert(to_upsert)

def query_notes(query: str, top_k: int = 3) -> list[str]:
    vector = embed_text([query])[0]
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
