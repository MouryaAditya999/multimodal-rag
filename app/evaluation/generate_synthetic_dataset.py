"""
Generate synthetic QA pairs using the local LLM.
Creates additional evaluation data beyond the 20 golden pairs.

Usage: python -m app.evaluation.generate_synthetic_dataset
"""

import json
import os
import sys
import time


def generate_synthetic_questions(chunks, num_pairs=30):
    """Use Mistral to generate QA pairs from retrieved chunks."""
    from app.llm.local_llm import llm

    prompt_template = """[INST] Given the context below, generate a question and its correct answer.
The question should test understanding of the content, not just copy-paste.

Context:
{context}

Output format:
Question: <question>
Answer: <answer>
[/INST]"""

    dataset = []
    used_contexts = set()

    for i, chunk in enumerate(chunks[:num_pairs * 2]):
        text = chunk[:800]
        if text in used_contexts:
            continue
        used_contexts.add(text)

        prompt = prompt_template.format(context=text)
        output = llm(prompt, max_tokens=200, temperature=0.7, stop=["</s>"])

        resp = output["choices"][0]["text"].strip()
        question = ""
        answer = ""
        for line in resp.split("\n"):
            if line.lower().startswith("question:"):
                question = line[len("question:"):].strip()
            elif line.lower().startswith("answer:"):
                answer = line[len("answer:"):].strip()

        if question and answer and len(question) > 10 and len(answer) > 5:
            dataset.append({
                "question": question,
                "ground_truth": answer,
                "source_doc": chunk.get("source", "synthetic"),
                "synthetic": True,
            })
            print(f"[{len(dataset)}] Q: {question[:60]}...")
            if len(dataset) >= num_pairs:
                break

    return dataset


def main():
    print("Generating synthetic QA pairs from ingested documents...")

    from app.config import BASE_DIR, REPORTS_DIR
    from app.vectorstore.chroma_store import get_collection

    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    chunks = [
        {"text": doc, "source": meta.get("source", "?"), "page": meta.get("page", "?")}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    print(f"Loaded {len(chunks)} chunks from ChromaDB")

    dataset = generate_synthetic_questions(chunks, num_pairs=30)

    output_path = os.path.join(REPORTS_DIR, "synthetic_dataset.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nGenerated {len(dataset)} synthetic QA pairs")
    print(f"Saved to {output_path}")

    # Combine with golden dataset and save full eval set
    from app.evaluation.evaluation_dataset import get_golden_dataset
    golden = get_golden_dataset()
    combined = golden + dataset
    combined_path = os.path.join(REPORTS_DIR, "full_evaluation_dataset.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined {len(golden)} golden + {len(dataset)} synthetic = {len(combined)} total")
    print(f"Full dataset saved to {combined_path}")


if __name__ == "__main__":
    main()
