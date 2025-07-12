from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()
