# rag_chatbot.py
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import re
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS  # Updated import: Use ddgs instead of duckduckgo_search

# Load environment variables
load_dotenv()

# Load RAG components
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)

# Groq LLM configuration
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=512 # reduced tokens for faster responses
)

system_prompt = (
    "You are a helpful KCAA assistant. Answer based ONLY on the provided context from local documents and web sources. "
    "If the question can't be answered from the context, say 'I don't have that information.' "
    "Use Markdown for readability (e.g., **bold**, *italics*, lists). \n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=prompt | llm)

def clean_filename(filename):
    """Clean and format filenames for better readability"""
    # Remove extensions and basic cleanup
    cleaned = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    # Handle specific patterns (e.g., remove trailing numbers if they're artifacts)
    cleaned = re.sub(r'\s*\d+$', '', cleaned)  # Remove trailing numbers like " 2"
    # Remove duplicates and extra spaces
    cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Capitalize properly
    return cleaned.strip().title()

def format_pages(pages):
    """Format a list of page numbers into compact ranges (e.g., '1-3, 5')"""
    if not pages:
        return ""
    pages = sorted(pages)
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = p
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)

def fetch_website_content(url):
    """Fetch and extract text content from a webpage using BS4"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract main text (customize selectors for KCAA site if needed)
        text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
        return text[:2000]  # Truncate to avoid token limits
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

def search_web(query, site=None, num_results=3):
    """Search web using DDGS (DuckDuckGo), optionally site-specific"""
    try:
        with DDGS() as ddgs:
            if site:
                query = f"{query} site:{site}"
            results = ddgs.text(query, max_results=num_results)
            snippets = [f"From {r['href']}: {r['body']}" for r in results]
            return '\n\n'.join(snippets)
    except Exception as e:
        return f"Error searching web: {str(e)}"

@st.cache_data
def ask_question(query):
    # Step 1: Local retrieval
    result = qa_chain.invoke({'input': query})
    local_answer = result['answer'].content
    
    # Group local pages by source
    source_pages = defaultdict(set)
    for doc in result['context']:
        source = doc.metadata.get('source')
        page = doc.metadata.get('page')
        if source and page is not None:
            source_pages[source].add(page + 1)  # 1-based
    
    # Build local sources
    local_sources = []
    for source in sorted(source_pages.keys()):
        cleaned = clean_filename(source)
        pages_str = format_pages(source_pages[source])
        if pages_str:
            cleaned += f" (Pages: {pages_str})"
        local_sources.append(cleaned)
    
    # Step 2: Check if local is sufficient; if not (or always for freshness), fetch web/social
    extended_context = ""
    web_sources = []
    
    # Fetch from KCAA website (specific pages or search)
    website_content = search_web(query, site="kcaa.or.ke", num_results=5)
    if "Error" not in website_content:
        extended_context += f"\n\nWeb Context from KCAA Site: {website_content}"
        web_sources.append("KCAA Website[](https://kcaa.or.ke/)")
    
    # Fetch from social media (site-specific searches for recent posts)
    social_sites = {
        "X (Twitter)": "twitter.com/CAA_Kenya",
        "LinkedIn": "linkedin.com/company/kenyacaa",
        "Facebook": "facebook.com/kcaake",
        "Instagram": "instagram.com/caa_kenya"
    }
    for platform, site in social_sites.items():
        social_content = search_web(query, site=site, num_results=3)
        if "Error" not in social_content:
            extended_context += f"\n\n{platform} Context: {social_content}"
            web_sources.append(f"{platform} ({site})")
    
    # Step 3: If extended context available, re-query LLM with combined context
    if extended_context:
        combined_context = f"{local_answer}\n\nAdditional Web/Social Context: {extended_context}"
        final_result = qa_chain.invoke({'input': query, 'context': combined_context})  # Override with extended
        answer = final_result['answer'].content
    else:
        answer = local_answer
    
    # Combine sources
    all_sources = local_sources + web_sources
    sources_md = "\n".join(f"• {source}" for source in sorted(set(all_sources)))  # Deduplicate
    
    return f"{answer}\n\n### Sources\n{sources_md}"

# Streamlit UI
st.title("KCAA Smart Assistant")
st.markdown("Ask about KCAA regulations, aviation info, and more. Responses are based on official documents, website, and social media.")

# Optional Sidebar
with st.sidebar:
    st.header("Options")
    st.write("Chat history is session-based. Refresh to clear.")
    st.info("Using: Groq + Llama 3.1 8B Instant with Web Integration")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Your question:"):
    # Add user message to history
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response with spinner
    with st.spinner("Searching KCAA regulations, website, and social..."):
        try:
            response = ask_question(user_input)
        except Exception as e:
            response = f"Sorry, I encountered an error. Please try again.\n\nError: {str(e)}"

    # Add assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})