from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel, Field 
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
tavily=TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str =Field(description="The url of the source")
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] =Field(default_factory=list, description="List of sources used to generate the answer")




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
agent=create_agent(model=llm, tools=tools,response_format=AgentResponse)
def main():
    print("Hello from langchain-course!")
    result=agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details.")})
    print(result)
if __name__ == "__main__":
    main()