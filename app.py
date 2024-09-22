from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

app = FastAPI()

# Initialize Groq API client
groq_api_key = "API_KEY"

# Function to load vector database
def load_vector_db(vector_db_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    return vector_db

# Initialize vector database
vector_db_path = "vector_db"
vector_db = load_vector_db(vector_db_path)
print("Vector database loaded successfully\n")

# Setup re-ranking
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192"
)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever(search_kwargs={"k": 5})
)

# Define the template for responding with document context
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

class Query(BaseModel):
    question: str

class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        # Retrieve and re-rank documents
        retrieved_docs = compression_retriever.get_relevant_documents(query.question)
        
        # Collect the links and areas (metadata) where the text was found
        doc_info = []
        for doc in retrieved_docs:
            page_number = doc.metadata.get('page', 'N/A')
            doc_link = doc.metadata.get('source', 'N/A')
            doc_info.append(DocumentInfo(
                page=str(page_number),
                link=doc_link,
                snippet=doc.page_content
            ))
        
        # Combine the content of the retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare the prompt with the retrieved context
        formatted_prompt = prompt.format(context=context, question=query.question)
        
        # Generate the response using Groq API
        response = llm.invoke(formatted_prompt)
        
        # Extract the content from the AIMessage
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return Response(answer=answer, retrieved_documents=doc_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)