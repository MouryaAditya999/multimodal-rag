"""Quick API smoke test - run this while the server is running."""
import requests
import json

BASE = "http://localhost:8000"

# Health check
r = requests.get(f"{BASE}/health")
print(f"Health: {r.json()}")

# Query
r = requests.post(
    f"{BASE}/query",
    json={"question": "What courses are taught in Semester VII?", "mode": "hybrid_rerank"},
)
data = r.json()
print(f"\nQuery answer (first 200 chars): {data['answer'][:200]}")
print(f"Contexts: {len(data['contexts'])} chunks")
print(f"Latencies: {data['latencies']}")
