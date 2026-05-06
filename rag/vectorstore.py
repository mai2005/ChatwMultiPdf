from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.config import EMBEDDING_MODEL, GOOGLE_API_KEY, FAISS_PATH
import streamlit as st

def get_vector_store(text_chunks, metadatas):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings, metadatas=metadatas)
    vector_store.save_local(FAISS_PATH)

@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
