
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DIR, EMBEDDING_MODEL_NAME, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD

_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

_bm25_retriever_cache = None # rebuilt whenever new docs are ingested


def get_vector_store():
    return Chroma(
        embedding_function=_embedding_model,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )


def get_all_documents():
    """Pulls every stored document back out of Chroma so BM25 can index them.
    BM25 isn't persisted like Chroma is, so it needs the full corpus in memory."""
    db = get_vector_store()
    data = db.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data['documents'], data['metadatas'])
    ]

def get_bm25_retriever(k=15, force_rebuild=False):
    global _bm25_retriever_cache
    if _bm25_retriever_cache is None or force_rebuild:
        docs = get_all_documents()
        _bm25_retriever_cache = BM25Retriever.from_documents(docs)
    _bm25_retriever_cache.k = k
    return _bm25_retriever_cache

def add_documents(documents, source_file: str = None):
    db = get_vector_store()
    db.add_documents(documents)
    print(f" Vector store updated at {CHROMA_DIR}")
    get_bm25_retriever(force_rebuild=True)
    return db


def get_retriever():
    db = get_vector_store()
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_K,
            "score_threshold": RETRIEVER_SCORE_THRESHOLD # Only return chunks with cosine similarity ≥ 0.3
        }
        # search_kwargs={
        #     "k": RETRIEVER_K, 
        #     "fetch_k": 10, 
        #     "lambda_mult" : 0.95 #0 = max diversity, 1 = max relevance
        #     }
    )

def delete_by_source(source_file: str):
    """Removes all previously-ingested chunks for this file, so re-ingesting
    the same file doesn't create duplicates."""
    db = get_vector_store()
    existing = db.get(where={"source_file": source_file})
    if existing["ids"]:
        db.delete(ids=existing["ids"])
        print(f"Removed {len(existing['ids'])} old chunks for {source_file}")
        get_bm25_retriever(force_rebuild=True)  # keep BM25 in sync