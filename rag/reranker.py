from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import RERANK_MODEL, GOOGLE_API_KEY

def rerank_ans(question, docs, top_k=3):
    llm = ChatGoogleGenerativeAI(model=RERANK_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY)
    context = ""
    for i, doc in enumerate(docs):
        context += f"\nDocument {i}:\n{doc.page_content}\n"

    prompt = f"""
You are a ranking system.

Given a query and a list of documents, select the {top_k} most relevant documents.

Query:
{question}

Documents:
{context}

Return ONLY the document numbers (e.g., 0,2,3)
"""
    
    response = llm.invoke(prompt).content

    try:
        selected_indices = [int(i.strip()) for i in response.split(",")]

    except:
        selected_indices = list(range(top_k))

    reranked_docs = [docs[i] for i in selected_indices if i < len(docs)]

    return reranked_docs
