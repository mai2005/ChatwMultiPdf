from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from utils.config import LLM_MODEL, GOOGLE_API_KEY

def get_conversation_chain():
    prompt_template = """
    You are a QA system.

    Rules:
    1. Only answer using the provided context
    2. If not found → say "NOT FOUND"
    3. Cite relevant parts

    Context:
    {context}

    Question:
    {question}

    Answer:
    - Final answer:
    - Supporting evidence (quote exact sentences from context):
"""
    
    model = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model=model, chain_type='stuff', prompt=prompt)
    return chain
