# build_knowledge_base.py (Updated for better chunking and structure)
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader  # Updated to preserve structure
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to your data folder
data_folder = 'data'

# Load all PDFs with better structure
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_folder, filename)
        loader = UnstructuredPDFLoader(file_path, mode="elements")  # Preserves headings/sections
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = filename  # Add filename as metadata for citations
        documents.extend(docs)

# Chunk the text (optimized: adjust size/overlap for better context)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)  # Increased for better context
chunks = text_splitter.split_documents(documents)

# Save chunks for BM25Retriever (for hybrid search)
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'  # Saves to disk for reuse
)

print(f"Knowledge base built with {len(chunks)} chunks.")