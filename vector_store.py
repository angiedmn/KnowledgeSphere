
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DIR, EMBEDDING_MODEL_NAME, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD

_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store():
    return Chroma(
        embedding_function=_embedding_model,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )


def add_documents(documents):
    db = get_vector_store()
    db.add_documents(documents)
    print(f" Vector store updated at {CHROMA_DIR}")
    from Hybrid_retriever import get_bm25_retriever  # local import avoids circular import
    get_bm25_retriever(force_rebuild=True)  # keep BM25 index in sync with new content

    return db


def get_retriever():
    db = get_vector_store()
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K, 
            "fetch_k": 10, 
            "lambda_mult" : 0.8 #0 = max diversity, 1 = max relevance
            }
    )

    
    