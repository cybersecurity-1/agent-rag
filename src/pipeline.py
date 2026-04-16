import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
# from langchain_chroma import Chroma  # ✅ replace old import
from langchain_community.vectorstores import Chroma

import streamlit as st

def get_hf_token() -> str:
    try:
        return st.secrets["HF_TOKEN"]
    except:
        return os.getenv("HF_TOKEN", "")

def get_embedding():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = get_hf_token()
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        encode_kwargs={"normalize_embeddings": True}
    )


# src/pipeline.py — remove the if/else check
def build_pipeline():
    embedding = get_embedding()

    loader = PyPDFLoader("data/hr_policy.pdf")
    documents = loader.load()
    print(f"✅ PDF loaded: {len(documents)} pages")

    chunks = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75
    ).split_documents(documents)
    print(f"✅ Chunks: {len(chunks)}")

    # ✅ Always rebuild — cloud storage is ephemeral
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB built")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]
    )

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return hybrid_retriever, cross_encoder, embedding


def create_dynamic_retriever(uploaded_file, embedding):
    if not uploaded_file:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    chunks = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    ).split_documents(docs)

    print(f"✅ Dynamic chunks: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_combined_docs(query, static_retriever, dynamic_retriever=None):
    static_docs = static_retriever.invoke(query)
    if dynamic_retriever:
        dynamic_docs = dynamic_retriever.invoke(query)
        return dynamic_docs + static_docs
    return static_docs