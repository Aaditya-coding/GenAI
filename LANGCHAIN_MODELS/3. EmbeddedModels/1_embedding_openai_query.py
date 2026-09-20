from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text", dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))