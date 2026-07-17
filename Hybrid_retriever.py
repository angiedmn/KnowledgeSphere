from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from vector_store import get_vector_store

_bm25_retriever_cache = None  # rebuilt whenever new docs are ingested

def get_all_documents():
    db = get_vector_store()
    data = db.get()  # pulls everything back out of Chroma
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data['documents'], data['metadatas'])
    ]

def get_bm25_retriever(k=15, force_rebuild=False):
    global _bm25_retriever_cache
    if _bm25_retriever_cache is None or force_rebuild:
        docs = get_all_documents()
        if not docs:
            raise ValueError("No documents in vector store yet — ingest documents before building BM25 index.")
        _bm25_retriever_cache = BM25Retriever.from_documents(docs)
    _bm25_retriever_cache.k = k
    return _bm25_retriever_cache

def refresh_bm25_index():
    """Rebuild the BM25 index from current Chroma contents. Call after ingesting new documents."""
    return get_bm25_retriever(force_rebuild=True)
