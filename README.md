## RAG-based PDF Question Answering System (with Reranking)

### Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDF documents and ask questions based on their content.  
The system retrieves relevant document chunks using vector search (FAISS) and improves retrieval quality using an additional LLM-based reranking step, before generating grounded answers with Google Gemini.

### System Architecture
```text
User Query
    ↓
Vector Search (FAISS - top 3)
    ↓
Reranking 
    ↓
LLM (Gemini)
    ↓
Final Answer
```

### Tech Stack
Frontend/UI: Streamlit  
LLM: Google Gemini (via LangChain)  
Embeddings: Gemini Embedding API  
Vector DB: FAISS  
Framework: LangChain

### Environment Setup
Create a .env file:  
GOOGLE_API_KEY=your_api_key_here

### Run the App
pip install -r requirements.txt  
streamlit run app.py
