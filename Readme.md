# KCAA Smart Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kc-chatbot.streamlit.app/ 

## Overview

KCAA Smart Assistant is an AI-powered chatbot designed to provide accurate and up-to-date information on Kenya Civil Aviation Authority (KCAA) regulations, aviation services, news, and more. It uses Retrieval-Augmented Generation (RAG) to query a local knowledge base built from official KCAA PDFs, supplemented with real-time web and social media searches for freshness.

Key technologies:
- **Frontend**: Streamlit for interactive chat UI.
- **Backend**: LangChain for RAG pipeline, HuggingFace embeddings for vector search, Chroma as vector database.
- **LLM**: Groq's Llama 3.1 8B model for fast, factual responses.
- **Web Integration**: DDGS (DuckDuckGo Search) for site-specific searches on KCAA website and social platforms (X/Twitter, LinkedIn, Facebook, Instagram).
- **Optimizations**: Conditional web fetching, parallel searches, and Markdown formatting for readability.

This is a prototype aimed at demonstrating quick, reliable aviation queries without heavy dependencies on paid APIs.

## Features

- **Local Knowledge Base**: Built from scraped/downloaded KCAA PDFs (e.g., regulations, notices) using `build_knowledge_base.py`.
- **Real-Time Augmentation**: Fetches from KCAA site and social media only when local info is insufficient, reducing latency.
- **User-Friendly UI**: Chat history, feedback buttons, clear cache/history options.
- **Security & Efficiency**: Input validation, reduced token limits, parallel web searches for speed.
- **Citations**: Detailed sources with PDF pages and web links for transparency.

## Installation

### Prerequisites
- Python 3.8+ (tested on 3.10)
- Virtual environment (recommended): `python -m venv .venv && source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
- API Keys: Sign up for [Groq](https://groq.com) and add `GROQ_API_KEY` to `.env`.

### Steps
1. Clone the repository:
   ```
   git clone https://https://github.com/ClementNdome/kcaa-demo.git
   cd kcaa-demo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Build the knowledge base (run once, or when adding new PDFs to `./data`):
   ```
   python build_knowledge_base.py
   ```

4. Create `.env` file with your keys:
   ```
   GROQ_API_KEY=your_groq_key_here
   ```

## Usage

1. Run the app:
   ```
   streamlit run app.py
   ```

2. Open in browser (defaults to http://localhost:8501).
3. Ask questions like "What are UAS regulations?" or "Latest KCAA news on aviation security."
   - Responses draw from local DB first; web/social if needed.
   - Use sidebar to clear cache/history.

### Example Queries
- Regulations: "Explain Category A UAS operations."
- News: "Recent KCAA events."
- Services: "How to get an Air Operator Certificate?"

## Project Structure

- `app.py`: Main Streamlit app and RAG logic.
- `build_knowledge_base.py`: Script to load PDFs, chunk, embed, and store in Chroma.
- `data/`: Folder for KCAA PDFs (add your scraped/downloaded files here).
- `chroma_db/`: Generated vector database.
- `chunks.pkl`: Pickled document chunks for potential hybrid search.
- `requirements.txt`: Dependencies list.
- `.env`: API keys (gitignore this!).
- `README.md`: This file.

## Performance Notes
- Latency: Optimized for 3-8 seconds per query (local hits faster).
- Limitations: Web searches use free DDGS (may have rate limits); no paid APIs for social (e.g., no Tweepy).
- Scaling: For production, consider cloud hosting (e.g., Streamlit Sharing) and caching web results.

## Contributing

Contributions welcome! Fork the repo, create a branch, and submit a PR.
- Issues: Report bugs or suggest features via GitHub Issues.
- Improvements: Add more PDFs, enhance prompts, or integrate advanced RAG (e.g., hybrid search).

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built as a prototype for efficient KCAA info access. For questions, contact [clementndome20@.com].
