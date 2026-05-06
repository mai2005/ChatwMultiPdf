from PyPDF2 import PdfReader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP

def get_pdf_text(pdf_docs):
    texts = []
    metadatas = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                metadatas.append({"source": pdf.name, "page": page_num})
    return texts, metadatas

def get_text_chunks(texts, metadatas):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks = []
    final_metadata = []

    for text, meta in zip(texts, metadatas):
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            final_chunks.append(chunk)
            new_meta = meta.copy()
            new_meta["chunk_id"] = i
            final_metadata.append(new_meta)

    return final_chunks, final_metadata