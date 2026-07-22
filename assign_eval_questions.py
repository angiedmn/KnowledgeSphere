"""

Run this once, check the printed assignment, then paste the results back
into eval_dataset.py's assigned_to fields (or just import and use
ASSIGNED_QUESTIONS directly from here instead).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from collections import defaultdict
from eval_dataset import eval_questions

TEAM_MEMBERS = ["aashna", "angel", "hiya"]  # EDIT with real names, in the order you want round-robin to start


def assign_questions(questions, team_members):
    # group by category first, so round-robin assignment happens WITHIN
    # each category rather than across the whole list — this is what
    # guarantees each person gets a mix of easy/hard/table/adversarial
    # questions instead of clumping by list order
    by_category = defaultdict(list)
    for q in questions:
        by_category[q["category"]].append(q)

    assigned = []
    for category, cat_questions in by_category.items():
        for i, q in enumerate(cat_questions):
            person = team_members[i % len(team_members)]
            q_copy = dict(q)
            q_copy["assigned_to"] = person
            assigned.append(q_copy)

    return assigned


def print_summary(assigned_questions, team_members):
    counts = defaultdict(lambda: defaultdict(int))
    for q in assigned_questions:
        counts[q["assigned_to"]][q["category"]] += 1

    print(f"{'Person':<12} | " + " | ".join(f"{cat:<16}" for cat in sorted({q['category'] for q in assigned_questions})) + " | Total")
    print("-" * 100)
    for person in team_members:
        cat_counts = counts[person]
        total = sum(cat_counts.values())
        row = " | ".join(f"{cat_counts.get(cat, 0):<16}" for cat in sorted({q['category'] for q in assigned_questions}))
        print(f"{person:<12} | {row} | {total}")


if __name__ == "__main__":
    assigned = assign_questions(eval_questions, TEAM_MEMBERS)

    print_summary(assigned, TEAM_MEMBERS)

    print("\n--- Per-person question lists ---")
    for person in TEAM_MEMBERS:
        person_questions = [q["question"] for q in assigned if q["assigned_to"] == person]
        print(f"\n{person} ({len(person_questions)} questions):")
        for q in person_questions:
            print(f"  - {q}")

    # write out a ready-to-use eval_dataset with assigned_to already filled in
    import json
    with open("eval_dataset_assigned.json", "w") as f:
        json.dump(assigned, f, indent=2)
    print("\nSaved full assignment to eval_dataset_assigned.json")