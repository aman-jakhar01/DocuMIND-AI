import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key,
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