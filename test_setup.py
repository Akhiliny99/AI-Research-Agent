import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from tavily import TavilyClient


llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
response = llm.invoke("Say hello in one word")
print("Groq OK:", response.content)


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
results = tavily.search("what is LangGraph", max_results=1)
print("Tavily OK:", results['results'][0]['title'])