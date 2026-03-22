import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google import genai
from operator import itemgetter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
load_dotenv()
print("Initializing components...")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", # This is the most reliable "alias"
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    output_dimensionality=1536
)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore=PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k":3})
prompt_template=ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer: """
)
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)
def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    Limitations:
    -Manual step-by-step execution
    -No built-in streaming support
    -No async support without additional code
    -Harder to compose with other chains
    -More verbose and error-prone
    """
    #Step 1: Retrieve relevant documents
    docs=retriever.invoke(query)
    #Steo 2: Format documents into context string
    context=format_docs(docs)
    # Step 3: Format the prompt with context and question
    messages=prompt_template.format_messages(context=context,question=query)
    #Step 4: Invoke LLM with the formatted messages
    response=llm.invoke(messages)
    return response.content

#Implementation 2:With LCEL (LangChain Expression Language_ -BETTER APPROACH)
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language)
    Returns a chain that can be invoked with {"question":"..."}
    
    Advantages over non-LCEL approach:
    Advantages over non-LCEL approach:
    -Declarative and composable: Easy to chain operations with pipe operator(|)
    -Built-in streaming: chain.stream() works out of the box
    -Built-in async: chain.invoke() and chain.astream() available
    -Batch processing: chain.batch() for multiple inputs
    -Type safety: Better integration with LangChain's type system
    -Less code: More concise and readable
    -Reusable: Chain can be saved, shared, and compose with other chains
    -Better debugging:LangChain provides beeter observability tools
    
    """
    retrieval_chain=(
       RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs) | prompt_template | llm |StrOutputParser()
    )

    return retrieval_chain
if __name__=="__main__":
    print("Retrieving...")
    query="what is Pinecone in machine learning?"
    #Option 0: Raw invocation without RAG  
    print("\n"+"="*70)
    print("Implementation 0: Raw LLM Invocation (NO RAG)")
    print("="*70)
    result_raw=llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
    #Option 1:Use implementation WITHOUT LCEL
    print("\n"+"="*70)
    print("Implementation 1:Without LCEl")
    print("="*70)
    result_without_lcel=retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)
    #Option 2: Use implementation WITH LCEL (Better Approach)
    print("\n"+"="*70)
    print("Implementation 2: With LCEL-better Aproach")
    print("="*70)
    print("\nWhy LCEL is better: ")
    print("-More concise and declarative")
    print("-Built-in streaming: chain.stream()")
    print("-Built-in async:chain.ainvoke()")
    print("-Easy to compose with other chains")
    print("-Better for production use")
    print("="*70)
    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question":query})
    print("\nAnswer")
    print(result_with_lcel)