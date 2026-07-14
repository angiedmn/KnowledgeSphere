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
    return db


def get_retriever():
    db = get_vector_store()
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD}
    )