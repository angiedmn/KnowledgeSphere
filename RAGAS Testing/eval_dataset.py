"""
Evaluation test set for RAGAS-based retrieval/answer evaluation.
Test document: "Attention Is All You Need" (Vaswani et al., 2017)

Covers a spread of difficulty levels as recommended for honest evaluation:
- Direct factual lookups (precision-heavy retrieval)
- Multi-concept synthesis questions (requires combining multiple chunks)
- Table-derived questions (numeric/comparison values from paper tables)
- Adversarial/negative cases (answer is NOT in the paper — tests hallucination resistance)
"""

eval_questions = [
    # --- Direct factual lookups ---
    {
        "question": "What is the name of the model architecture introduced in this paper?",
        "ground_truth": "The Transformer, a model architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "category": "direct_factual",
    },
    {
        "question": "How many attention heads does the base Transformer model use?",
        "ground_truth": "The base model uses 8 attention heads (h = 8).",
        "category": "direct_factual",
    },
    {
        "question": "What is the dimensionality of the model, d_model, used in the base Transformer?",
        "ground_truth": "d_model = 512 for the base model.",
        "category": "direct_factual",
    },
    {
        "question": "Why do the authors scale the dot products by 1/sqrt(d_k) in the attention mechanism?",
        "ground_truth": "To counteract the effect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.",
        "category": "direct_factual",
    },
    {
        "question": "What type of positional encoding does the Transformer use?",
        "ground_truth": "Sine and cosine functions of different frequencies are used as positional encodings, added to the input embeddings.",
        "category": "direct_factual",
    },
    {
        "question": "What optimizer was used to train the Transformer models?",
        "ground_truth": "The Adam optimizer, with beta_1 = 0.9, beta_2 = 0.98, and epsilon = 10^-9.",
        "category": "direct_factual",
    },
    {
        "question": "What regularization techniques are used during training?",
        "ground_truth": "Residual dropout applied to the output of each sub-layer and to the sums of embeddings and positional encodings, and label smoothing with a value of 0.1.",
        "category": "direct_factual",
    },

    # --- Multi-concept synthesis ---
    {
        "question": "How does the encoder-decoder attention mechanism in the Transformer differ from self-attention in the encoder?",
        "ground_truth": "In encoder-decoder attention layers, queries come from the previous decoder layer, while keys and values come from the output of the encoder, allowing every position in the decoder to attend over all positions in the input sequence. In contrast, self-attention layers in the encoder have queries, keys, and values all coming from the same place — the output of the previous encoder layer.",
        "category": "multi_concept",
    },
    {
        "question": "What is the purpose of masking in the decoder's self-attention layers, and how is it implemented?",
        "ground_truth": "Masking prevents positions from attending to subsequent positions, preserving the auto-regressive property. It is implemented by masking out (setting to negative infinity) all values in the input of the softmax which correspond to illegal connections, combined with offsetting output embeddings by one position.",
        "category": "multi_concept",
    },
    {
        "question": "Why do the authors argue self-attention layers are advantageous compared to recurrent and convolutional layers?",
        "ground_truth": "Self-attention layers have a constant number of sequentially executed operations, connect all positions with a constant number of operations (versus O(n) for recurrent layers), and have lower total computational complexity per layer when sequence length n is smaller than representation dimensionality d — enabling more parallelization and better handling of long-range dependencies.",
        "category": "multi_concept",
    },
    {
        "question": "How do the multi-head attention and the feed-forward layers work together within a single encoder layer?",
        "ground_truth": "Each encoder layer has two sub-layers: a multi-head self-attention mechanism, followed by a position-wise fully connected feed-forward network. A residual connection is employed around each sub-layer, followed by layer normalization.",
        "category": "multi_concept",
    },

    # --- Table-derived ---
    {
        "question": "What BLEU score did the Transformer (big) model achieve on the WMT 2014 English-to-German translation task?",
        "ground_truth": "28.4 BLEU.",
        "category": "table_derived",
    },
    {
        "question": "What BLEU score did the Transformer (big) model achieve on the WMT 2014 English-to-French translation task?",
        "ground_truth": "41.8 BLEU.",
        "category": "table_derived",
    },
    {
        "question": "According to the training cost comparison table, approximately how many floating point operations (FLOPs) were used to train the Transformer (big) model?",
        "ground_truth": "Approximately 2.3 x 10^19 FLOPs.",
        "category": "table_derived",
    },
    {
        "question": "In the model variations table, what happens to perplexity when the number of attention heads is reduced to 1?",
        "ground_truth": "Perplexity worsens (increases) compared to the base configuration with 8 heads — single-head attention is 0.9 BLEU worse than the best setting.",
        "category": "table_derived",
    },

    # --- Adversarial / negative cases (answer should NOT be found in this paper) ---
    {
        "question": "What dataset was used to pretrain BERT in this paper?",
        "ground_truth": "Not applicable — this paper does not discuss BERT or its pretraining data; BERT was published after this paper.",
        "category": "adversarial",
    },
    {
        "question": "What is the GPU memory requirement for fine-tuning the Transformer on a single consumer GPU?",
        "ground_truth": "Not applicable — the paper does not report consumer-GPU memory requirements; it reports training on 8 NVIDIA P100 GPUs.",
        "category": "adversarial",
    },
    {
        "question": "How does the Transformer's performance compare to GPT-4 on reasoning benchmarks?",
        "ground_truth": "Not applicable — this paper predates GPT-4 and does not compare against it.",
        "category": "adversarial",
    },
    {
        "question": "What learning rate warmup schedule is used, and for how many steps?",
        "ground_truth": "The learning rate increases linearly for the first warmup_steps training steps (warmup_steps = 4000), then decreases proportionally to the inverse square root of the step number.",
        "category": "direct_factual",
    },
    {
        "question": "What is the computational complexity per layer of a self-attention layer compared to a recurrent layer, in terms of sequence length n and representation dimension d?",
        "ground_truth": "Self-attention: O(n^2 * d) per layer. Recurrent: O(n * d^2) per layer.",
        "category": "table_derived",
    },
]

