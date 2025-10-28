# #==== THIS code outputs good results yes, but they not human friendly enough. , the below code does better formatting
# # rag_chatbot.py
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq  # Changed from OllamaLLM
# import os
# from dotenv import load_dotenv
# from langchain_classic.chains import create_retrieval_chain
# from langchain_core.prompts import ChatPromptTemplate

# # Load environment variables
# load_dotenv()

# # Load RAG components
# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)

# # Replace Ollama with Groq
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",  # Best equivalent to phi3:mini
#     temperature=0.1,  # Keep responses factual
#     api_key=os.getenv("GROQ_API_KEY"),  # Your API key from .env
#     max_tokens=1024  # Optional: control response length
# )

# system_prompt = (
#     "You are a helpful KCAA assistant. Answer based ONLY on the provided context. "
#     "If the question can't be answered from the context, say 'I don't have that information.' "
#     "Use Markdown for readability (e.g., **bold**, *italics*, lists). "
#     "Cite sources at the end as a bullet list.\n\n"
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
# qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=prompt | llm)

# # Query function
# @st.cache_data
# def ask_question(query):
#     result = qa_chain.invoke({'input': query})
#     answer = result['answer']
#     sources = set(doc.metadata['source'] for doc in result['context'])
#     sources_md = "\n".join(f"- {source}" for source in sources)
#     return f"{answer}\n\n**Sources:**\n{sources_md}"

# # Streamlit UI
# st.title("KCAA Smart Assistant")
# st.markdown("Ask about KCAA regulations, aviation info, and more. Responses are based on official documents.")

# # Optional Sidebar
# with st.sidebar:
#     st.header("Options")
#     st.write("Chat history is session-based. Refresh to clear.")
#     # Show which model we're using
#     st.info(f"Using: Groq + Llama 3.1 8B Instant")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input
# if user_input := st.chat_input("Your question:"):
#     # Add user message to history
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # Generate response with spinner
#     with st.spinner("Thinking..."):
#         try:
#             response = ask_question(user_input)
#         except Exception as e:
#             response = f"Sorry, I encountered an error: {str(e)}"

#     # Add assistant message
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})


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
    max_tokens=1024
)

system_prompt = (
    "You are a helpful KCAA assistant. Answer based ONLY on the provided context. "
    "If the question can't be answered from the context, say 'I don't have that information.' "
    "Use Markdown for readability (e.g., **bold**, *italics*, lists). "
    "Cite sources at the end as a bullet list.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=prompt | llm)

def clean_filename(filename):
    """Clean and format filenames for better readability"""
    # Remove file extensions and clean up formatting
    cleaned = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    # Remove duplicate words and clean up legal notice formatting
    cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned)  # Remove duplicate words
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove extra spaces
    return cleaned.strip().title()

@st.cache_data
def ask_question(query):
    result = qa_chain.invoke({'input': query})
    answer = result['answer']
    
    # Extract and clean source names
    sources = set(doc.metadata['source'] for doc in result['context'])
    cleaned_sources = [clean_filename(source) for source in sources]
    
    # Create formatted sources section
    sources_md = "\n".join(f"• {source}" for source in sorted(cleaned_sources))
    
    return f"{answer}\n\n**Sources:**\n{sources_md}"

# Streamlit UI
st.title("KCAA Smart Assistant")
st.markdown("Ask about KCAA regulations, aviation info, and more. Responses are based on official documents.")

# Optional Sidebar
with st.sidebar:
    st.header("Options")
    st.write("Chat history is session-based. Refresh to clear.")
    st.info("Using: Groq + Llama 3.1 8B Instant")
    
    # Display usage stats (optional)
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
    with st.spinner("Searching KCAA regulations..."):
        try:
            response = ask_question(user_input)
        except Exception as e:
            response = f"Sorry, I encountered an error. Please try again.\n\nError: {str(e)}"

    # Add assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})