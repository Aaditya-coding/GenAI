from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

result = llm.invoke("What is the capital of Norway?")

print(result)