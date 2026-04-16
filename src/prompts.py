import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st

# ✅ Works both locally and on Streamlit Cloud
def get_api_key(key: str) -> str:
    try:
        return st.secrets[key]
    except:
        return os.getenv(key, "")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer for a HR policy chatbot.

Your job:
- Convert the user question into a clear, specific search query
- Preserve important keywords (e.g., leave policy, probation, notice period)
- Use simple keywords (no full sentences)
- Max 10 words

Rules:
- Output ONLY the query
- No explanation, no quotes

Conversation History:
{history}

User Question:
{question}

Search Query:
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are an expert HR assistant.

Instructions:
1. Answer ONLY from the provided context
2. If multiple rules exist, combine them clearly
3. If answer is not found, say: "Not mentioned in the policy"
4. Be precise (include numbers, conditions, limits)
5. Avoid vague answers

Conversation:
{history}

Retrieved Context:
{context}

User Question:
{question}

Final Answer:
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()


# import os
# from langchain_groq import ChatGroq

# # Only LLM needed — no prompts/chains
# # Agent handles prompting internally via SystemMessage
# llm = ChatGroq(
#     model_name="openai/gpt-oss-20b",
#     groq_api_key=os.getenv("GROQ_API_KEY")
# )