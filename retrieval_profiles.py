def build_retrieval_profile(directness: float, semanticness: float) -> dict:
    """
    directness:   0.0 = Diverse ... 1.0 = Direct
    semanticness: 0.0 = Keyword ... 1.0 = Semantic
    """
    k = round(3 + (1 - directness) * 7)              # Direct: 3 chunks -> Diverse: 10 chunks
    num_variations = round((1 - directness) * 4)     # Direct: 0 rephrasings -> Diverse: 4
    lambda_mult = 0.3 + directness * 0.65             # Diverse: 0.3 (varied) -> Direct: 0.95 (relevance-focused)

    vector_weight = 0.15 + semanticness * 0.7          # Keyword: 0.15 -> Semantic: 0.85
    keyword_weight = 1.0 - vector_weight

    return {
        "k": k,
        "search_type": "mmr",
        "lambda_mult": lambda_mult,
        "num_variations": num_variations,
        "vector_weight": vector_weight,
        "keyword_weight": keyword_weight,
    }


DEFAULT_PROFILE = build_retrieval_profile(directness=0.5, semanticness=0.5)  # "Balanced"