import asyncio
import os 
import ssl
from typing import Any,Dict,List
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

async def main():
    """Main asyn function to orchestrate the entire process"""
    load_dotenv()


if __name__=="__main__":
    asyncio.run(main())