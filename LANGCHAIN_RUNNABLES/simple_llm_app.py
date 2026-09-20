from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Define the input
topic = input('Enter a topic')

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the LLM directly
blog_title = llm.invoke(formatted_prompt)

# Print the output
print("Generated Blog Title:", blog_title)