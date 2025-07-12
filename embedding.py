from gradio_client import Client

# Connect to your deployed Space
client = Client("TheTharindu/meta_note_embedding_generator")

def embed_text(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = client.predict(text, api_name="/predict")
        embeddings.append(result)
    return embeddings
