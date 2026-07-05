from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

print("Embedding model loaded.")


def embed_text(text, is_query=False):
    """Encode text with nomic task prefix for better retrieval."""
    prefix = "search_query: " if is_query else "search_document: "
    return model.encode(
        prefix + text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def generate_embeddings(chunks):
    embedded_chunks = []

    for chunk in chunks:
        embedding = embed_text(chunk["text"], is_query=False)

        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": embedding.tolist(),
            }
        )

    return embedded_chunks
