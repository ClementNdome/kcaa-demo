# build_knowledge_base.py
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to your data folder
data_folder = 'data'

# Load all PDFs
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_folder, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = filename  # Add filename as metadata for citations
        documents.extend(docs)

# Chunk the text (optimized: larger chunks for better context)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Save chunks for BM25Retriever
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# Store in Chroma (simplified, no tenant/database)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'  # Saves to disk for reuse
)

print(f"Knowledge base built with {len(chunks)} chunks.")