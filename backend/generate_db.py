import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from web_links import web_links

# Load environment variables
from dotenv import load_dotenv
load_dotenv("../secrets.env")

# Initialize embeddings
embeddings = OllamaEmbeddings(model="llama3.2")

# Load web documents
web_loader = WebBaseLoader(web_paths=web_links)
web_docs = web_loader.load()

# Load and process Excel file
excel_df = pd.read_excel("../Active Learning Repo.xlsx")
excel_string = excel_df.to_string(index=False)
excel_docs = [Document(page_content=excel_string, metadata={"source": "Excel file"})]

# Combine documents
docs = web_docs + excel_docs

# clean docs
for doc in docs:
    cleaned_content = " ".join(doc.page_content.split())
    doc.page_content = cleaned_content

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

# Create FAISS vector store
vector_store = FAISS.from_documents(final_documents, embeddings)

# Save vector store to disk
vector_store.save_local("faiss_index")
print("FAISS database saved to disk.")