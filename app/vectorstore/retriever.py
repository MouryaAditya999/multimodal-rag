from app.config import TOP_K, TOP_K_HYBRID, TOP_K_RERANK
from app.vectorstore.chroma_store import get_collection


def get_context_for_query(query, mode="hybrid"):
    if mode == "dense":
        return _dense_retrieve(query)
    elif mode == "bm25":
        return _bm25_retrieve(query)
    elif mode == "hybrid":
        return _hybrid_retrieve(query)
    elif mode == "hybrid_rerank":
        return _hybrid_rerank_retrieve(query)
    else:
        return _dense_retrieve(query)


def _dense_retrieve(query):
    from app.embeddings.embedder import embed_text
    from app.vectorstore.chroma_store import get_collection

    collection = get_collection()
    query_embedding = embed_text(query, is_query=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    return results


def _bm25_retrieve(query):
    from app.retrieval.bm25_retriever import retrieve as bm25_retrieve_fn

    return bm25_retrieve_fn(query, top_k=TOP_K)


def _hybrid_retrieve(query):
    from app.retrieval.hybrid_retriever import retrieve as hybrid_retrieve_fn

    return hybrid_retrieve_fn(query, top_k=TOP_K_HYBRID)


def _hybrid_rerank_retrieve(query):
    from app.retrieval.hybrid_retriever import retrieve_with_rerank

    return retrieve_with_rerank(query, top_k=TOP_K_HYBRID, rerank_top_k=TOP_K_RERANK)


def retrieve(query, top_k=TOP_K):
    collection = get_collection()
    query_embedding = embed_text(query, is_query=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results
