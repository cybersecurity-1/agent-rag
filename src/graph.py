import operator
import streamlit as st
from typing import TypedDict, List, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.helpers import rerank_docs, format_docs, format_history
from src.prompts import rewrite_chain, answer_chain
from src.pipeline import get_combined_docs


class RAGState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    rewritten: str
    context: str
    answer: str


def make_graph(retriever, cross_encoder):

    def rewrite_node(state: RAGState):
        print("\n" + "="*50)
        print("📝 NODE: REWRITE")
        print(f"   Original question: {state['question']}")

        rewritten = rewrite_chain.invoke({
            "question": state["question"],
            "history": format_history(state["messages"])
        })

        print(f"   Rewritten query:   {rewritten}")
        print("="*50)
        return {"rewritten": rewritten}

    def retrieve_node(state: RAGState):
        print("\n" + "="*50)
        print("🔍 NODE: RETRIEVE")
        print(f"   Query: {state['rewritten']}")

        dynamic = st.session_state.get("dynamic_retriever")
        print(f"   Dynamic retriever: {'✅ Yes' if dynamic else '❌ No (using static)'}")

        docs = get_combined_docs(
            state["rewritten"],
            retriever,
            dynamic
        )

        print(f"   Docs retrieved: {len(docs)}")
     
        return {"context": docs}

    def rerank_node(state: RAGState):
        print("\n" + "="*50)
        print("🏆 NODE: RERANK")
        print(f"   Docs before rerank: {len(state['context'])}")

        final_docs = rerank_docs(
            state["rewritten"],
            state["context"],
            cross_encoder,
            top_k=5
        )

        print(f"   Docs after rerank:  {len(final_docs)}")
       
        return {"context": format_docs(final_docs)}

    def answer_node(state: RAGState):
        print("\n" + "="*50)
        print("💬 NODE: ANSWER")
        print(f"   Question: {state['question']}")
        print(f"   Context length: {len(state['context'])} chars")
        print(f"   Memory messages: {len(state['messages'])}")

        answer = answer_chain.invoke({
            "question": state["question"],
            "context": state["context"],
            "history": format_history(state["messages"])
        })

        print(f"   Answer: {answer[:100]}...")
        print("="*50)
        return {
            "answer": answer,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer)
            ]
        }

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", END)

    return graph.compile()