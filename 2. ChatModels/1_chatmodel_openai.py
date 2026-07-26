from langchain_ollama import ChatOllama
import time

model1 = ChatOllama(model="qwen3:8b", temperature=0.2, max_completion_tokens=10)
start = time.time()

result1 = model1.invoke("Write a 5 line poem on cricket.")


print(result1.content)
print(f"\n Time taken: {time.time()-start:.2f} seconds")