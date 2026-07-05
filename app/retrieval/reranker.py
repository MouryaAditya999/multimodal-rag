from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL


class Reranker:
    _instance = None

    def __init__(self):
        self.model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            print(f"Loading reranker: {RERANKER_MODEL} ...")
            cls._instance.model = CrossEncoder(RERANKER_MODEL)
            print("Reranker loaded.")
        return cls._instance


def rerank(query, documents, top_k=5):
    reranker = Reranker.get_instance()
    pairs = [(query, doc) for doc in documents]
    scores = reranker.model.predict(pairs)
    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
