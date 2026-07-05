import argparse
import glob
import os
import sys


def run_ingestion():
    from app.ingestion.document_loader import load_document, load_multimodal_content
    from app.chunking.chunker import chunk_documents
    from app.embeddings.embedder import generate_embeddings, model as embed_model
    from app.vectorstore.chroma_store import store_chunks, store_image_chunks, client, get_collection, get_image_collection
    from app.config import DOCUMENTS_DIR

    print("=" * 60)
    print("INGESTION FLOW")
    print("=" * 60)

    try:
        client.delete_collection("text_chunks")
        print("Cleared existing text collection.")
    except Exception:
        pass
    try:
        client.delete_collection("image_chunks")
        print("Cleared existing image collection.")
    except Exception:
        pass

    get_collection()
    get_image_collection()

    supported_extensions = ["*.pdf", "*.docx", "*.txt", "*.md", "*.csv", "*.xlsx", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(DOCUMENTS_DIR, ext)))

    if not files:
        print(f"No documents found in {DOCUMENTS_DIR}.")
        return

    print(f"Found {len(files)} files to ingest:")
    for f in files:
        print(f" - {os.path.basename(f)}")

    all_chunks = []
    all_image_chunks = []
    chunk_id_offset = 0

    for file_path in files:
        fname = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        print(f"\nProcessing: {fname} ...")

        # Try multimodal extraction first (gets text + tables + images)
        content = load_multimodal_content(file_path)

        # Process text and table content
        text_and_table_pages = []
        text_and_table_pages.extend(content["texts"])
        # Tables are already text-serialized
        for t in content["tables"]:
            text_and_table_pages.append(t)

        if text_and_table_pages:
            chunks = chunk_documents(text_and_table_pages, start_id=chunk_id_offset + 1)
            all_chunks.extend(chunks)
            chunk_id_offset += len(chunks)
            print(f"  Text/Table chunks: {len(chunks)}")

        # Process images
        for img in content["images"]:
            all_image_chunks.append(img)
        if content["images"]:
            print(f"  Images extracted: {len(content['images'])}")

        # Fallback: plain text extraction if multimodal returned nothing
        if not content["texts"] and not content["tables"] and not content["images"]:
            pages = load_document(file_path)
            if pages:
                chunks = chunk_documents(pages)
                all_chunks.extend(chunks)
                print(f"  Text pages: {len(pages)}, chunks: {len(chunks)}")

    if not all_chunks and not all_image_chunks:
        print("No content extracted from documents.")
        return

    # Store text/table chunks
    if all_chunks:
        print(f"\nTotal text/table chunks: {len(all_chunks)}")
        print("Generating embeddings...")
        embedded_chunks = generate_embeddings(all_chunks)
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        print("Storing in ChromaDB...")
        store_chunks(all_chunks, embeddings)

    # Store image chunks (with captions)
    if all_image_chunks:
        print(f"\nTotal images: {len(all_image_chunks)}")
        print("Generating caption embeddings for search...")
        for img in all_image_chunks:
            caption = img.get("caption", "")
            if caption:
                emb = embed_model.encode(
                    "search_document: " + caption,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                img["text_embedding"] = emb.tolist()
            else:
                img["text_embedding"] = None
        print("Storing image chunks in ChromaDB...")
        store_image_chunks(all_image_chunks)

    print("\nIngestion completed successfully!")


def run_query(question, mode="hybrid"):
    from app.llm.local_llm import generate_answer
    from app.utils import truncate_contexts
    from app.vectorstore.retriever import get_context_for_query

    print(f"\nRunning Query: '{question}'")
    print(f"Retrieval mode: {mode}\n")

    if mode == "dense":
        results = get_context_for_query(question, mode="dense")
        documents = truncate_contexts(results["documents"][0])
        metadatas = results["metadatas"][0][: len(documents)]
    else:
        results = get_context_for_query(question, mode=mode)
        documents = truncate_contexts(results["documents"])
        metadatas = results["metadatas"][: len(documents)]

    def _safe(text): return text.encode('utf-8', 'replace').decode('utf-8')
    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        source = meta.get("source", "Unknown")
        page = meta.get("page", "Unknown")
        print(f"[{i+1}] Source: {source} (Page {page})")
        print(f"    Snippet: {_safe(doc[:200])}...")
        print("-" * 60)

    context = "\n\n".join(documents)

    print("\nGenerating answer with local LLM...")
    answer = generate_answer(context, question)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(_safe(answer))
    print("=" * 60)


def run_evaluate(sample_size=None, mode="dense"):
    from app.evaluation.evaluate import run_evaluation

    run_evaluation(sample_size=sample_size, mode=mode)


def run_ablation(sample_size=None):
    from app.evaluation.evaluate import run_ablation

    run_ablation(sample_size=sample_size)


def run_generate_synthetic():
    from app.evaluation.generate_synthetic_dataset import main as gen_main
    gen_main()


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("ingest", help="Ingest files from data/documents into ChromaDB")

    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="The question to ask. If omitted, starts an interactive shell.",
    )
    query_parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval mode (default: hybrid_rerank)",
    )

    eval_parser = subparsers.add_parser(
        "evaluate", help="Run offline RAGAS evaluation"
    )
    eval_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of QA pairs to evaluate (default: all 20)",
    )
    eval_parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
        default="dense",
        help="Retrieval mode for evaluation (default: dense for baseline)",
    )

    ablation_parser = subparsers.add_parser(
        "ablation", help="Run full ablation study comparing all retrieval modes"
    )
    ablation_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of QA pairs per experiment (default: all 20)",
    )

    subparsers.add_parser(
        "generate-synthetic", help="Generate synthetic QA pairs using local LLM (Track A dataset)"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion()
    elif args.command == "query":
        if args.question:
            run_query(args.question, mode=args.mode)
        else:
            print("Entering interactive RAG shell. Type 'exit' or 'quit' to exit.")
            while True:
                try:
                    q = input("\nAsk a question: ").strip()
                    if not q:
                        continue
                    if q.lower() in ["exit", "quit"]:
                        break
                    run_query(q, mode=args.mode)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
    elif args.command == "evaluate":
        run_evaluate(sample_size=args.sample, mode=args.mode)
    elif args.command == "ablation":
        run_ablation(sample_size=args.sample)
    elif args.command == "generate-synthetic":
        run_generate_synthetic()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
