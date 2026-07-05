from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(pages, start_id=1):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    chunk_id = start_id

    for page in pages:
        split_chunks = splitter.split_text(page["text"])

        for chunk in split_chunks:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "source": page["source"],
                    "format": page.get("format", "unknown"),
                    "modality": page.get("modality", "text"),
                    "text": chunk,
                }
            )
            chunk_id += 1

    return chunks
