"""
Run this once all teammates have sent their eval_results_<name>.json files.

Merges every eval_results_*.json in this folder into one combined dataset,
then scores it with RAGAS (judge model wired to Groq, not OpenAI).


"""

import json
import glob
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME


def merge_results():
    files = sorted(glob.glob("eval_results_*.json"))
    files = [f for f in files if f != "eval_results_combined.json"]  # don't re-merge a previous merge

    if not files:
        raise FileNotFoundError(
            "No eval_results_<name>.json files found. Make sure teammates' files "
            "are in this folder before running this script."
        )

    all_results = []
    for filepath in files:
        with open(filepath) as f:
            person_results = json.load(f)
            all_results.extend(person_results)
        print(f"Loaded {len(person_results)} results from {filepath}")

    print(f"\nMerged {len(all_results)} total results from {len(files)} files")

    with open("eval_results_combined.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved merged results to eval_results_combined.json")

    return all_results


def score_with_ragas(all_results):
    judge_llm = LangchainLLMWrapper(ChatGroq(model=LLM_MODEL_NAME, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    answer_relevancy_metric = AnswerRelevancy(strictness=1) 
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in all_results
    ])

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy_metric, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1),  
    )
    print("\n=== RAGAS Scores (overall) ===")
    print(scores)

    return scores


def print_latency_summary(all_results):
    latencies = [r["latency_ms"] for r in all_results if "latency_ms" in r]
    if not latencies:
        return
    avg = sum(latencies) / len(latencies)
    print(f"\n=== Latency ===")
    print(f"Average: {avg:.0f} ms | Min: {min(latencies):.0f} ms | Max: {max(latencies):.0f} ms")


def print_category_breakdown(all_results):
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r.get("category", "uncategorized")].append(r)

    print("\n=== Category Breakdown ===")
    for cat, items in sorted(by_category.items()):
        print(f"{cat}: {len(items)} questions")


def print_assignment_breakdown(all_results):
    by_person = defaultdict(list)
    for r in all_results:
        by_person[r.get("assigned_to", "unknown")].append(r)

    print("\n=== Contribution Breakdown ===")
    for person, items in sorted(by_person.items()):
        print(f"{person}: {len(items)} questions")


if __name__ == "__main__":
    all_results = merge_results()
    print_assignment_breakdown(all_results)
    print_category_breakdown(all_results)
    print_latency_summary(all_results)
    scores = score_with_ragas(all_results)