from app.config import RRF_WEIGHT_BM25, RRF_WEIGHT_DENSE, TOP_K_HYBRID, TOP_K_RERANK
from app.embeddings.embedder import embed_text
from app.retrieval.bm25_retriever import retrieve as bm25_retrieve
from app.retrieval.bm25_retriever import load_corpus_from_chromadb
from app.vectorstore.chroma_store import get_collection


def _rrf(ranked_lists, weights, k=60):
    score_map = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked):
            score_map[item_id] = score_map.get(item_id, 0.0) + weight / (k + rank + 1)
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


def retrieve(query, top_k=TOP_K_HYBRID):
    # Ensure BM25 corpus is loaded
    load_corpus_from_chromadb()

    # Dense retrieval via ChromaDB
    collection = get_collection()
    query_embedding = embed_text(query, is_query=True).tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # Sparse retrieval via BM25
    bm25_results = bm25_retrieve(query, top_k=top_k)

    # Get all document texts for lookup
    all_data = collection.get(include=["documents", "metadatas"])
    doc_map = {}
    for i, doc_id in enumerate(all_data["ids"]):
        doc_map[doc_id] = {
            "text": all_data["documents"][i],
            "metadata": all_data["metadatas"][i],
        }
    for doc_id, meta in zip(
        bm25_results["ids"], bm25_results["metadatas"]
    ):
        if doc_id not in doc_map:
            doc_map[doc_id] = {"text": "", "metadata": meta}

    # RRF fusion
    dense_ids = dense_results["ids"][0] if dense_results["ids"] else []
    bm25_ids = bm25_results["ids"]
    fused = _rrf(
        [dense_ids, bm25_ids],
        [RRF_WEIGHT_DENSE, RRF_WEIGHT_BM25],
    )

    # Build final results
    results = {"documents": [], "metadatas": [], "ids": [], "scores": []}
    seen = set()
    for doc_id, score in fused:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in doc_map:
            results["documents"].append(doc_map[doc_id]["text"])
            results["metadatas"].append(doc_map[doc_id]["metadata"])
            results["ids"].append(doc_id)
            results["scores"].append(round(score, 4))
            if len(results["documents"]) >= top_k:
                break

    return results


def retrieve_with_rerank(query, top_k=TOP_K_HYBRID, rerank_top_k=TOP_K_RERANK):
    from app.retrieval.reranker import Reranker

    results = retrieve(query, top_k=top_k)
    if not results["documents"]:
        return results

    reranker = Reranker.get_instance()
    pairs = [(query, doc) for doc in results["documents"]]
    rerank_scores = reranker.model.predict(pairs)

    scored = list(
        zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
            rerank_scores,
        )
    )
    scored.sort(key=lambda x: x[3], reverse=True)

    reranked = {
        "documents": [],
        "metadatas": [],
        "ids": [],
        "scores": [],
    }
    for doc, meta, doc_id, score in scored[:rerank_top_k]:
        reranked["documents"].append(doc)
        reranked["metadatas"].append(meta)
        reranked["ids"].append(doc_id)
        reranked["scores"].append(float(score))

    return reranked
