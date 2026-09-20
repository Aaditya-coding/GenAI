from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)

print(llm.invoke('hi'))

from langchain.agents import create_agent

# Step 2 : Create the ReAct agent manually with the pulled prompt
agent = create_agent(
    model = llm,
    tools = [search_tool],
)

# Step 4 : Invoke
response = agent.invoke({
    "messages": [
    {
        "role": "user",
        "content": "3 ways to reach goa from delhi" 
    }   
    ]
})

print(response)