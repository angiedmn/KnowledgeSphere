
from typing import List
from collections import defaultdict
from pydantic import BaseModel
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME


from typing import List
from collections import defaultdict
from pydantic import BaseModel
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME

class QueryVariations(BaseModel):
    queries: List[str]

def generate_query_variations(query: str, num_variations: int = 3) -> List[str]:
    if num_variations <= 0:          # NEW — lets "Direct" mode skip this LLM call entirely
        return []

    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
    llm_with_structure = llm.with_structured_output(QueryVariations)
    
    prompt = f"""Generate {num_variations} different variations of this query that would help retrieve relevant documents:

Original query: {query}

Return {num_variations} alternative queries that rephrase or approach the same question from different angles."""
    
    response = llm_with_structure.invoke(prompt)
    return response.queries


def reciprocal_rank_fusion(chunk_lists, weights=None, k=60):   
    if weights is None:
        weights = [1.0] * len(chunk_lists)   

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    
    for chunks, weight in zip(chunk_lists, weights):   
        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content
            all_unique_chunks[chunk_content] = chunk
            rrf_scores[chunk_content] += weight * (1 / (k + position))   
            
    sorted_chunks = sorted(
        [(all_unique_chunks[c], score) for c, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_chunks


def multi_query_retrieve(query: str, retriever, num_variations: int = 3, top_k: int = 5):
    """
    UNCHANGED — kept exactly as-is. Still vector-only, no weighting.
    """
    print("\n--- Generating Query Variations ---")
    variations = generate_query_variations(query, num_variations)
    all_queries = [query] + variations
    
    for i, q in enumerate(all_queries):
        print(f"Query {i+1}: {q}")
        
    print("\n--- Retrieving and Fusing Documents ---")
    all_retrieval_results = [retriever.invoke(q) for q in all_queries]
    
    fused = reciprocal_rank_fusion(all_retrieval_results)
    final_docs = [doc for doc, score in fused[:top_k]]
    
    print(f"Successfully fused and retrieved top {len(final_docs)} documents!")
    return final_docs


def hybrid_retrieve(query: str, vector_retriever, bm25_retriever, vector_weight=0.6, keyword_weight=0.4):
    
    """One query -> vector results + keyword results -> weighted RRF-fused per-query list."""
    vector_docs = vector_retriever.invoke(query)

    if bm25_retriever is None:
        keyword_docs = []  
    else:
        keyword_docs = bm25_retriever.invoke(query)

    fused = reciprocal_rank_fusion(
        [vector_docs, keyword_docs],
        weights=[vector_weight, keyword_weight]   
    )
    return [doc for doc, score in fused]


def multi_query_hybrid_retrieve(query: str, vector_retriever, bm25_retriever, num_variations: int = 3,
                                 top_k: int = 5, vector_weight=0.6, keyword_weight=0.4):
    """
    Full pipeline: query -> multiple phrasings -> weighted hybrid (vector + BM25)
    retrieve each -> final RRF across all of them.
    """
    print("\n--- Generating Query Variations ---")
    variations = generate_query_variations(query, num_variations)
    all_queries = [query] + variations

    for i, q in enumerate(all_queries):
        print(f"Query {i+1}: {q}")

    print("\n--- Hybrid Retrieving and Fusing Documents ---")
    per_query_results = [
        hybrid_retrieve(q, vector_retriever, bm25_retriever, vector_weight, keyword_weight)   # CHANGED
        for q in all_queries
    ]

    fused = reciprocal_rank_fusion(per_query_results)   
    final_docs = [doc for doc, score in fused[:top_k]]

    print(f"Successfully fused and retrieved top {len(final_docs)} documents!")
    return final_docs

# class QueryVariations(BaseModel):
#     queries: List[str]

# def generate_query_variations(query: str, num_variations: int = 3) -> List[str]:
#     llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
#     llm_with_structure = llm.with_structured_output(QueryVariations)
    
#     prompt = f"""Generate {num_variations} different variations of this query that would help retrieve relevant documents:

# Original query: {query}

# Return {num_variations} alternative queries that rephrase or approach the same question from different angles."""
    
#     response = llm_with_structure.invoke(prompt)
#     return response.queries

# def reciprocal_rank_fusion(chunk_lists, k=60):
#     rrf_scores = defaultdict(float)
#     all_unique_chunks = {}
    
#     for chunks in chunk_lists:
#         for position, chunk in enumerate(chunks, 1):
#             chunk_content = chunk.page_content
#             all_unique_chunks[chunk_content] = chunk
#             rrf_scores[chunk_content] += 1 / (k + position)
            
#     sorted_chunks = sorted(
#         [(all_unique_chunks[c], score) for c, score in rrf_scores.items()],
#         key=lambda x: x[1],
#         reverse=True
#     )
#     return sorted_chunks


# def multi_query_retrieve(query: str, retriever, num_variations: int = 3, top_k: int = 5):
#     """
#     Expands `query` into multiple phrasings, retrieves for each (plus the
#     original), fuses all result lists via RRF, and returns the top_k fused
#     Documents.
#     """
#     print("\n--- Generating Query Variations ---")
#     variations = generate_query_variations(query, num_variations)
#     all_queries = [query] + variations
    
#     for i, q in enumerate(all_queries):
#         print(f"Query {i+1}: {q}")
        
#     print("\n--- Retrieving and Fusing Documents ---")
#     all_retrieval_results = [retriever.invoke(q) for q in all_queries]
    
#     fused = reciprocal_rank_fusion(all_retrieval_results)
#     final_docs = [doc for doc, score in fused[:top_k]]
    
#     print(f"Successfully fused and retrieved top {len(final_docs)} documents!")
#     return final_docs

# def hybrid_retrieve(query: str, vector_retriever, bm25_retriever):
#     """One query -> vector results + keyword results -> RRF-fused per-query list."""
#     vector_docs = vector_retriever.invoke(query)

#     if bm25_retriever is None:
#         keyword_docs = []  # no keyword index built yet, fall back to vector-only
#     else:
#         keyword_docs = bm25_retriever.invoke(query)

#     fused = reciprocal_rank_fusion([vector_docs, keyword_docs])
#     return [doc for doc, score in fused]

# def multi_query_hybrid_retrieve(query: str, vector_retriever, bm25_retriever, num_variations: int = 3, top_k: int = 5):
#     """
#     Full pipeline: query -> multiple phrasings -> hybrid (vector + BM25)
#     retrieve each -> final RRF across all of them.
#     """
#     print("\n--- Generating Query Variations ---")
#     variations = generate_query_variations(query, num_variations)
#     all_queries = [query] + variations

#     for i, q in enumerate(all_queries):
#         print(f"Query {i+1}: {q}")

#     print("\n--- Hybrid Retrieving and Fusing Documents ---")
#     per_query_results = [hybrid_retrieve(q, vector_retriever, bm25_retriever) for q in all_queries]

#     fused = reciprocal_rank_fusion(per_query_results)
#     final_docs = [doc for doc, score in fused[:top_k]]

#     print(f"Successfully fused and retrieved top {len(final_docs)} documents!")
#     return final_docs
























