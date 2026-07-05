from app.embeddings.embedder import embed_text
from app.vectorstore.chroma_store import get_image_collection


def retrieve_images(query, top_k=5):
    """Retrieve relevant images by querying their captions."""
    collection = get_image_collection()
    if collection.count() == 0:
        return {"documents": [], "metadatas": [], "ids": []}

    query_embedding = embed_text(query, is_query=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results
