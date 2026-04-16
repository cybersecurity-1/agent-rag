from langchain_core.messages import HumanMessage


def rerank_docs(query, docs, cross_encoder, top_k=3):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(messages):
    if not messages:
        return "No previous conversation."
    history = []
    for m in messages[-6:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history.append(f"{role}: {m.content}")
    return "\n".join(history)