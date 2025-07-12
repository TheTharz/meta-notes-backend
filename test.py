from gradio_client import Client

client = Client("TheTharindu/meta_note_embedding_generator")
embedding = client.predict("test", api_name="/predict")
print("Embedding dimension:", len(embedding))
