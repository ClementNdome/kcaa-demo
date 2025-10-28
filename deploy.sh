#!/bin/bash

echo "Building RAG Chatbot Docker image..."
docker build -t rag-chatbot .

echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Warning: Ollama may not be running. Start it with: ollama serve"
fi

echo "Starting container..."
docker run -d \
    --name rag-chatbot \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/chroma_db:/app/chroma_db \
    rag-chatbot

echo "RAG Chatbot is running at http://localhost:8501"