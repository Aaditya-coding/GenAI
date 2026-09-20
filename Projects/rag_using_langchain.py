import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# Step 1a - Indexing(Document Ingestion)

video_id = "LPZh9BOjkQs"  # Only the ID, not full URL
try:
    # If you don't care which language, this returns the "best" one
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b - Indexing(Text Splitting)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1c & 1d - Indexing(Embedding Generation and Storing in Vector Store)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 2 - Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":8}) 

# Step 3 - Augmentation

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    max_new_tokens=512,
    temperature=0.2,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provived transcript context.
    If the context is insufficient, just say you don't know.
    
    {context}
    Question: {question}""",
    input_variables={'context','question'}
)

question = str(input("Ask me something...\n"))
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question":question})
# 4. Generation

answer = llm.invoke(final_prompt)
print(answer)

# Building a Chain

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# parallel_chain.invoke('who is Demis?')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the video')