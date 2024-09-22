import os
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                yield page_num, text

def process_pdf(pdf_path, chunk_size=1800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    documents = []
    for page_num, page_text in extract_text_from_pdf(pdf_path):
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path,
                    "page": page_num
                }
            )
            documents.append(doc)
    
    print(f"Number of chunks created from {pdf_path}: {len(documents)}")
    return documents

def process_multiple_pdfs(pdf_paths):
    all_documents = []
    for pdf_path in pdf_paths:
        documents = process_pdf(pdf_path)
        all_documents.extend(documents)
    return all_documents

def process_and_store(pdf_paths, vector_db_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    
    all_documents = process_multiple_pdfs(pdf_paths)
    
    # Check if the vector database already exists
    if os.path.exists(vector_db_path):
        print("Existing vector database found. Updating with new documents...")
        vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        vector_db.add_documents(all_documents)
    else:
        print("Creating new vector database...")
        vector_db = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=vector_db_path
        )
    
    vector_db.persist()
    print(f"All chunks from {len(pdf_paths)} PDFs processed and stored in the vector database")

def main():
    pdf_paths = [
        "NCERT-Class-12-Physics-Part-1.pdf",
        # "NCERT-Class-12-Physics-Part-2.pdf"
    ]
    vector_db_path = "vector_db3"
    
    process_and_store(pdf_paths, vector_db_path)

if __name__ == "__main__":
    main()