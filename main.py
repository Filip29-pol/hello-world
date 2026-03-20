from dotenv import load_dotenv
import os
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
tavily=TavilyClient()
@tool
def search(query: str) -> str:
    """Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching the web for: {query}")
    return tavily.search(query=query)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", # Or "models/gemini-1.5-flash"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0)
tools=[search]
agent=create_agent(model=llm, tools=tools)
def main():
    print("Hello from langchain-course!")
    result=agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details.")})
    print(result)
if __name__ == "__main__":
    main()