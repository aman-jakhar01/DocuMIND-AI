import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Local environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Cloud
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
)


def ask_document(question, context):

    prompt = f"""
You are DocuMind AI, an intelligent document analysis assistant.

Answer the user's question using ONLY the information provided
in the document context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the uploaded document."

Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content