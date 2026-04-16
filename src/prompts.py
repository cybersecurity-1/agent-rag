import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the question as a short search query (max 8 words).
Output ONLY the query. No explanation. No preamble. No quotes.

History:
{history}

Question: {question}
Query:""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant with memory of the conversation.

Previous conversation:
{history}

Context from document:
{context}

Current question: {question}

Answer:""")

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