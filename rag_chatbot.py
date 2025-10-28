# rag_chatbot.py
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load RAG components (same as before)
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
llm = OllamaLLM(model='phi3:mini') 

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

# Query function (same as before)
@st.cache_data  # Cache for speed (Streamlit-specific)
def ask_question(query):
    result = qa_chain.invoke({'input': query})
    answer = result['answer']
    sources = set(doc.metadata['source'] for doc in result['context'])
    sources_md = "\n".join(f"- {source}" for source in sources)
    return f"{answer}\n\n**Sources:**\n{sources_md}"

# Streamlit UI
st.title("KCAA Smart Assistant")
st.markdown("Ask about KCAA regulations, aviation info, and more. Responses are based on official documents.")

# Optional Sidebar (example of Streamlit's flexibility)
with st.sidebar:
    st.header("Options")
    st.write("Chat history is session-based. Refresh to clear.")
    # Add more: e.g., model selector, feedback form

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

    # Generate response with spinner (for UX)
    with st.spinner("Thinking..."):
        response = ask_question(user_input)

    # Add assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Optional: Log query (for your analytics)
    # with open("queries.log", "a") as f:
    #     f.write(f"Query: {user_input}\nResponse: {response}\n\n")