#latency.py
import json
import pandas as pd

with open("eval_results_combined.json") as f:
    results = json.load(f)

df = pd.DataFrame(results)

print("=== Overall Latency ===")
print(f"Average: {df['latency_ms'].mean():.0f} ms")
print(f"Min:     {df['latency_ms'].min():.0f} ms")
print(f"Max:     {df['latency_ms'].max():.0f} ms")

print("\n=== Latency by Category ===")
print(df.groupby("category")["latency_ms"].mean().round(0))

print("\n=== Latency by Team Member (different machines/networks) ===")
print(df.groupby("assigned_to")["latency_ms"].mean().round(0))