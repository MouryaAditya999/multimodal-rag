import chromadb

from app.config import CHROMA_DIR

client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    """Get or create the text chunks collection."""
    return client.get_or_create_collection(name="text_chunks")


def get_image_collection():
    """Get or create the image chunks collection."""
    return client.get_or_create_collection(name="image_chunks")


def store_chunks(chunks, embeddings):
    """Store chunks and embeddings in ChromaDB."""
    collection = get_collection()
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        documents.append(chunk["text"])
        metadatas.append(
            {
                "page": chunk["page"],
                "source": chunk["source"],
                "format": chunk.get("format", "unknown"),
                "modality": chunk.get("modality", "text"),
            }
        )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\nStored {len(ids)} chunks successfully!")


def store_image_chunks(image_chunks):
    """Store image metadata + caption embeddings in ChromaDB."""
    collection = get_image_collection()
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, img in enumerate(image_chunks):
        caption = img.get("caption", "")
        doc_text = caption if caption else f"Image from {img['source']} page {img.get('page','?')}"
        ids.append(f"img_{i}")
        documents.append(doc_text)
        metadatas.append(
            {
                "source": img["source"],
                "page": img.get("page", 1),
                "format": img.get("format", "unknown"),
                "modality": "image",
                "image_path": img["image_path"],
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }
        )
        if img.get("text_embedding") is not None:
            embeddings.append(img["text_embedding"])

    if embeddings:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    else:
        # Store without embeddings if none available
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"Stored {len(ids)} image chunks successfully!")
