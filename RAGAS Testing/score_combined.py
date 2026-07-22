
from dotenv import load_dotenv
load_dotenv()

import json
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI   # NEW — judge only, not used anywhere else
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME   # only the embedding model comes from config — no Groq import here at all

with open("eval_results_combined.json") as f:
    all_results = json.load(f)

print(f"Scoring {len(all_results)} combined results")

# Judge LLM: Gemini — completely separate from your main app's Groq-based LLM_MODEL_NAME
judge_llm = judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
answer_relevancy_metric = AnswerRelevancy(strictness=1)

dataset = Dataset.from_list([
    {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"], "ground_truth": r["ground_truth"]}
    for r in all_results
])

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy_metric, context_precision],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=RunConfig(max_workers=2),  # Gemini's free tier handles a bit more concurrency than Groq's did
)

print("\n=== RAGAS Scores (overall, combined 20 questions) ===")
print(scores)

scores_df = scores.to_pandas()
scores_df["category"] = [r["category"] for r in all_results]
scores_df["assigned_to"] = [r.get("assigned_to", "unknown") for r in all_results]
scores_df["latency_ms"] = [r.get("latency_ms", None) for r in all_results]
scores_df.to_csv("final_scores_combined.csv", index=False)
print("\nSaved to final_scores_combined.csv")

print("\n=== By category ===")
print(scores_df.groupby("category")[["faithfulness", "answer_relevancy", "context_precision"]].mean())