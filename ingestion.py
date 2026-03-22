import os
from dotenv import load_dotenv


load_dotenv()

from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_pinecone import PineconeVectorStore

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", # This is the most reliable "alias"
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    output_dimensionality=1536
)

    print("Ingesting to Pinecone...")
    
    PineconeVectorStore.from_documents(
        texts, 
        embeddings, 
        index_name=os.environ.get('INDEX_NAME')
    )
    print("Finish")