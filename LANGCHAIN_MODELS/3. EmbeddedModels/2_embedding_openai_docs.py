from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text", dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Berlin is the capital of Germany",
    "Paris is the capital of France"
]
result = embedding.embed_documents(documents)

print(str(result))