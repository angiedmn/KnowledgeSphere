
from dotenv import load_dotenv
load_dotenv()

import sys
import time
import json

import conversational_RAG
from conversational_RAG import ask_question
from retrieval_profiles import build_retrieval_profile

with open("eval_dataset_assigned.json") as f:
    eval_questions = json.load(f)

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME


TEAM_MEMBER = None

if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    TEAM_MEMBER = sys.argv[1]

my_questions = [q for q in eval_questions if q["assigned_to"] == TEAM_MEMBER]
print(f"Running {len(my_questions)} questions assigned to {TEAM_MEMBER}")

eval_profile = build_retrieval_profile(directness=1.0, semanticness=0.5)


def get_ragas_judge():
    judge_llm = LangchainLLMWrapper(ChatGroq(model=LLM_MODEL_NAME, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    return judge_llm, judge_embeddings


def run_pipeline_on_test_set(questions):
    results = []
    for item in questions:
        conversational_RAG.chat_history = []  # reset — each eval question is independent
        print(f"Running: {item['question']}")
        start = time.perf_counter()
        answer, source_pages, contexts = ask_question(
            item["question"], retrieval_profile=eval_profile, return_contexts=True
        )
        latency_ms = (time.perf_counter() - start) * 1000

        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "category": item["category"],
            "assigned_to": item["assigned_to"],
            "latency_ms": latency_ms,
        })

   
    output_file = f"eval_results_{TEAM_MEMBER}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {output_file}")

    return results


def score_with_ragas(results):
    judge_llm, judge_embeddings = get_ragas_judge()

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ])

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    print(scores)

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    print(f"\nAverage latency: {avg_latency:.0f} ms")

    return scores


if __name__ == "__main__":
    results = run_pipeline_on_test_set(my_questions)
    scores = score_with_ragas(results)