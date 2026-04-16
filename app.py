import os
from dotenv import load_dotenv
load_dotenv()

port = int(os.environ.get("PORT", 10000))

import streamlit as st
from src.pipeline import build_pipeline, create_dynamic_retriever
from src.graph import make_graph
from tavily import TavilyClient


@st.cache_resource
def load_system():
    retriever, cross_encoder, embedding = build_pipeline()
    graph = make_graph(retriever, cross_encoder)
    return graph, embedding


def tavily_web_search(query: str) -> list:
    """Search the web using Tavily API and return results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        st.error("❌ TAVILY_API_KEY not found in environment variables.")
        return []
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    return response.get("results", [])


st.set_page_config(page_title="HR Policy RAG", layout="wide")
st.title("🧠 HR Policy Chatbot")
st.caption("Advanced RAG + LangGraph Memory")

graph, embedding = load_system()

# ✅ Clear chat button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.graph_state = {"messages": []}
        st.session_state.display_messages = []
        st.rerun()

# PDF upload
uploaded_file = st.file_uploader("📎 Upload extra PDF (optional)", type="pdf")

if "dynamic_retriever" not in st.session_state:
    st.session_state.dynamic_retriever = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if uploaded_file:
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.dynamic_retriever = create_dynamic_retriever(
            uploaded_file, embedding
        )
        st.session_state.uploaded_filename = uploaded_file.name
        st.success(f"✅ Loaded: {uploaded_file.name}")
    else:
        st.info(f"📄 Using: {uploaded_file.name}")
else:
    st.session_state.dynamic_retriever = None
    st.session_state.uploaded_filename = None

# Chat state
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "web_search_mode" not in st.session_state:
    st.session_state.web_search_mode = False

# Display history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── 🌐 Web Search Toggle — tight above the chat input ───────────────────────
st.markdown("""
<style>
/* Remove gap between toggle row and chat input */
div[data-testid="stHorizontalBlock"] { margin-bottom: 1.5rem !important; }
section[data-testid="stBottom"] { margin-top: 0 !important; padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

search_col, _ = st.columns([1, 9])
with search_col:
    web_toggle = st.button(
        "🌐" if not st.session_state.web_search_mode else "🌐✅",
        key="web_search_toggle",
        help="Web Search via Tavily — ON: searches web | OFF: uses documents",
        use_container_width=True,
    )
    if web_toggle:
        st.session_state.web_search_mode = not st.session_state.web_search_mode
        st.rerun()

# ─── Chat Input ───────────────────────────────────────────────────────────────
query = st.chat_input("Ask something about the document...")

if query:
    st.session_state.display_messages.append({
        "role": "user", "content": query
    })
    with st.chat_message("user"):
        st.write(query)

    # ── Web Search Mode ──────────────────────────────────────────────────────
    if st.session_state.web_search_mode:
        st.session_state.web_search_mode = False   # auto-reset after one search
        with st.spinner("🌐 Searching the web via Tavily..."):
            web_results = tavily_web_search(query)

        if web_results:
            answer_parts = [f"### 🌐 Web Search Results for: *{query}*\n"]
            for i, r in enumerate(web_results, 1):
                title   = r.get("title", "No title")
                url     = r.get("url", "")
                content = r.get("content", "")[:300]
                answer_parts.append(
                    f"**{i}. [{title}]({url})**\n{content}...\n"
                )
            web_answer = "\n".join(answer_parts)
        else:
            web_answer = "⚠️ No web results found for your query."

        with st.chat_message("assistant"):
            st.markdown(web_answer)

        st.session_state.display_messages.append({
            "role": "assistant", "content": web_answer
        })

    # ── RAG Mode (default) ───────────────────────────────────────────────────
    else:
        with st.spinner("Thinking..."):
            result = graph.invoke({   # previous memory and current question passed into graph
                **st.session_state.graph_state,
                "question": query,
            })
            st.session_state.graph_state = result
            answer = result["answer"]

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("🔍 Debug Info — All Nodes"):

                # Node 1 — Rewrite
                st.markdown("### 📝 Node 1: Rewrite")
                st.markdown(f"**Original:** `{query}`")
                st.markdown(f"**Rewritten:** `{result['rewritten']}`")
                st.divider()

                # Node 2 — Retrieve
                st.markdown("### 🔍 Node 2: Retrieve")
                dynamic_active = st.session_state.dynamic_retriever is not None
                st.markdown(f"**Dynamic retriever:** {'✅ Active' if dynamic_active else '❌ Not active'}")
                st.divider()

                # Node 3 — Rerank
                st.markdown("### 🏆 Node 3: Rerank")
                st.markdown("Top 3 docs after CrossEncoder reranking:")
                context_preview = result['context'][:500] if result['context'] else "No context"
                st.text(context_preview + "...")
                st.divider()

                # Node 4 — Answer
                st.markdown("### 💬 Node 4: Answer")
                st.markdown(f"**Memory:** {len(result['messages'])} messages stored")
                st.markdown(f"**Context length:** {len(result['context'])} chars used")

        st.session_state.display_messages.append({
            "role": "assistant", "content": answer
        })
