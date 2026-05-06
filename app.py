import streamlit as st

from rag.ingestion import get_pdf_text, get_text_chunks
from rag.vectorstore import get_vector_store, load_vector_store
from rag.reranker import rerank_ans
from rag.qa import get_conversation_chain
from utils.config import FAISS_PATH 

def user_input(user_question):

    db = load_vector_store()
    init_docs = db.similarity_search(user_question, k=10)
    docs = rerank_ans(user_question, init_docs, top_k=3)

    chain = get_conversation_chain()

    response = chain.invoke({'input_documents': docs, 'question': user_question})

    st.subheader("Reply")
    st.write(response['output_text'])

    with st.expander("Retrieved context (before rerank)"):
        for i, doc in enumerate(init_docs):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content[:100])
            st.write(f"Source: {doc.metadata}")
            st.divider()
    
    with st.expander("Retrieved context (after rerank)"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content[:100])
            st.write(f"Source: {doc.metadata}")
            st.divider()

def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Enter your question here...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the Submit button", accept_multiple_files=True)
        if st.button("Submit"):
            if not pdf_docs:
                st.warning("Please upload at least one pdf.")
                return
            with st.spinner("Processing..."):
                raw_text, metadatas = get_pdf_text(pdf_docs)
                text_chunks, metadata = get_text_chunks(raw_text, metadatas)
                get_vector_store(text_chunks, metadata)
                st.success("Done!")

if __name__ == '__main__':
    main()
