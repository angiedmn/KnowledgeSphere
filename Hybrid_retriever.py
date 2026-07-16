
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL_NAME, RETRIEVER_K
from vector_store import get_vector_store, get_all_documents

_bm25_retriever = None  # built once, cached


def _get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        all_docs = get_all_documents()
        _bm25_retriever = BM25Retriever.from_documents(all_docs)
        _bm25_retriever.k = RETRIEVER_K
    return _bm25_retriever


def refresh_bm25_index():
    """Call this after new documents are ingested, so BM25 sees the new content."""
    global _bm25_retriever
    _bm25_retriever = None


def generate_query_variants(question, n=3):
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        """Generate {n} different rephrasings of the question below, to help
        retrieve relevant documents. Return ONLY the questions, one per line,
        no numbering, no extra text.

        Question: {question}"""
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "n": n})
    variants = [q.strip() for q in result.split("\n") if q.strip()]
    return [question] + variants


def reciprocal_rank_fusion(ranked_lists, k=60, top_n=None):
    scores = {}
    doc_map = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = doc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        fused = fused[:top_n]
    return [doc_map[key] for key, _ in fused]


def hybrid_search(query, k=None, n_variants=3):
    """Multi-query + Hybrid (vector + BM25) + RRF, all in one call."""
    k = k or RETRIEVER_K
    db = get_vector_store()
    bm25 = _get_bm25_retriever()

    queries = generate_query_variants(query, n=n_variants)

    all_ranked_lists = []
    for q in queries:
        vector_results = db.similarity_search(q, k=k)
        bm25_results = bm25.invoke(q)
        all_ranked_lists.append(vector_results)
        all_ranked_lists.append(bm25_results)

    return reciprocal_rank_fusion(all_ranked_lists, top_n=k)