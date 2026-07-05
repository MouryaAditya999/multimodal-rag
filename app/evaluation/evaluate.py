import gc
import json
import os
import time
from datetime import datetime

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from app.config import EMBEDDING_MODEL, REPORTS_DIR, TOP_K, TOP_K_RERANK
from app.evaluation.evaluation_dataset import get_golden_dataset
from app.llm import local_llm
from app.llm.eval_wrapper import LocalMistralLLM
from app.llm.local_llm import generate_answer
from app.utils import truncate_contexts
from app.vectorstore.retriever import get_context_for_query

TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
    "latency_first_token_sec": 3.0,
    "latency_full_response_sec": 20.0,
}


def _build_ragas_models(reuse_llm=None):
    llm_instance = reuse_llm
    if llm_instance is None:
        llm_instance = LocalMistralLLM()._llm
    judge_llm = LangchainLLMWrapper(LocalMistralLLM(llm_instance=llm_instance))
    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    return judge_llm, ragas_embeddings


def _run_rag_pipeline(question, mode="dense"):
    t0 = time.perf_counter()

    if mode == "dense":
        results = get_context_for_query(question, mode="dense")
        documents = truncate_contexts(results["documents"][0])
    else:
        results = get_context_for_query(question, mode=mode)
        documents = truncate_contexts(results["documents"])

    context = "\n\n".join(documents)
    t_retrieval = time.perf_counter()
    answer = generate_answer(context, question)
    t_end = time.perf_counter()

    return {
        "contexts": documents,
        "answer": answer,
        "retrieval_latency_sec": round(t_retrieval - t0, 3),
        "generation_latency_sec": round(t_end - t_retrieval, 3),
        "total_latency_sec": round(t_end - t0, 3),
    }


def run_evaluation(sample_size=None, mode="dense", output_path=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f"eval_{mode}.json")

    dataset = get_golden_dataset()
    if sample_size:
        dataset = dataset[:sample_size]

    mode_label = mode.replace("_", " ").title()
    print("=" * 60)
    print(f"EVALUATION: {mode_label}")
    print("=" * 60)
    print(f"Evaluating {len(dataset)} QA pairs...\n")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    latency_records = []

    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['question'][:60]}...")
        result = _run_rag_pipeline(item["question"], mode=mode)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        latency_records.append(
            {
                "question": item["question"],
                "retrieval_sec": result["retrieval_latency_sec"],
                "generation_sec": result["generation_latency_sec"],
                "total_sec": result["total_latency_sec"],
            }
        )

    print(f"\nRunning RAGAS metrics with local Mistral judge ({mode})...")
    del local_llm.llm
    gc.collect()

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    run_config = RunConfig(timeout=600, max_workers=1, max_retries=2)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    scores = {}

    for metric in metrics:
        print(f"  Computing {metric.name}...")
        judge_llm, ragas_embeddings = _build_ragas_models()
        try:
            result = evaluate(
                eval_dataset,
                metrics=[metric],
                llm=judge_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
                raise_exceptions=True,
            )
            scores[metric.name] = float(result._repr_dict[metric.name])
            print(f"    -> {metric.name}: {scores[metric.name]:.4f}")
        except Exception as exc:
            print(f"    -> {metric.name} failed: {exc}")
            scores[metric.name] = None
        finally:
            del judge_llm, ragas_embeddings
            gc.collect()

    avg_retrieval = sum(r["retrieval_sec"] for r in latency_records) / len(
        latency_records
    )
    avg_generation = sum(r["generation_sec"] for r in latency_records) / len(
        latency_records
    )
    avg_total = sum(r["total_sec"] for r in latency_records) / len(latency_records)

    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": f"{mode}_retrieval",
        "retrieval_mode": mode,
        "top_k": TOP_K,
        "num_samples": len(dataset),
        "ragas_scores": scores,
        "latency": {
            "avg_retrieval_sec": round(avg_retrieval, 3),
            "avg_generation_sec": round(avg_generation, 3),
            "avg_total_sec": round(avg_total, 3),
            "per_query": latency_records,
        },
        "targets": TARGETS,
        "targets_met": {
            "faithfulness": (scores.get("faithfulness") or 0)
            >= TARGETS["faithfulness"],
            "answer_relevancy": (scores.get("answer_relevancy") or 0)
            >= TARGETS["answer_relevancy"],
            "context_precision": (scores.get("context_precision") or 0)
            >= TARGETS["context_precision"],
            "context_recall": (scores.get("context_recall") or 0)
            >= TARGETS["context_recall"],
            "latency_total": avg_total <= TARGETS["latency_full_response_sec"],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_report(report, output_path)
    return report


def run_ablation(sample_size=None):
    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    results = {}
    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"# ABLATION: {mode}")
        print(f"{'#'*60}")
        results[mode] = run_evaluation(sample_size=sample_size, mode=mode)
    _print_ablation_report(results)
    return results


def _print_report(report, output_path):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in report["ragas_scores"].items():
        target = report["targets"].get(metric)
        status = ""
        if score is None:
            print(f"  {metric}: FAILED")
            continue
        if target is not None:
            met = score >= target
            status = " PASS" if met else " BELOW TARGET"
        print(f"  {metric}: {score:.4f}{status}")
    print("\nLatency:")
    print(f"  Avg retrieval:   {report['latency']['avg_retrieval_sec']:.3f}s")
    print(f"  Avg generation:  {report['latency']['avg_generation_sec']:.3f}s")
    print(f"  Avg total:       {report['latency']['avg_total_sec']:.3f}s")
    print(f"\nReport saved to: {output_path}")
    print("=" * 60)


def _print_ablation_report(results):
    print("\n" + "=" * 60)
    print("ABLATION STUDY - SUMMARY")
    print("=" * 60)
    header = f"{'Mode':<20} {'Faith':<10} {'Relev':<10} {'Prec':<10} {'Recall':<10} {'Latency':<10}"
    print(header)
    print("-" * 70)
    for mode, report in results.items():
        s = report["ragas_scores"]
        lat = report["latency"]["avg_total_sec"]
        print(
            f"{mode:<20} {s.get('faithfulness', 0):<10.4f} "
            f"{s.get('answer_relevancy', 0):<10.4f} "
            f"{s.get('context_precision', 0):<10.4f} "
            f"{s.get('context_recall', 0):<10.4f} "
            f"{lat:<10.3f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
