import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from a local .env file.
load_dotenv()


def get_groq_api_key():
    """Get the Groq API key from Streamlit Cloud Secrets or environment."""

    # Try Streamlit Cloud Secrets first.
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key and str(api_key).strip():
            return str(api_key).strip()
    except Exception:
        # Streamlit Secrets may not be configured locally.
        pass

    # Fall back to environment variables / local .env.
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    return None


GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Add it to "
        "Streamlit Cloud Settings > Secrets, or configure it in your local .env file."
    )
    st.stop()


# Initialize the Groq language model.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


def ask_document(question, context):
    """Answer questions using only the supplied document context."""

    prompt = f"""
You are DocuMind AI, an intelligent document analysis assistant.

Rules:
- Answer using ONLY the information in the document context.
- Do not invent facts or use outside knowledge.
- If the answer is not available in the context, respond exactly:
  "I couldn't find this information in the uploaded document."
- Give clear, direct answers.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content
