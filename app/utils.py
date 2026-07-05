"""Shared context formatting helpers."""

from app.config import TOP_K


def truncate_contexts(contexts, max_chars_per_chunk=400, max_chunks=4):
    """Trim retrieved contexts to fit local LLM context window."""
    return [c[:max_chars_per_chunk] for c in contexts[:max_chunks]]


def build_context_from_results(results):
    """Extract and truncate documents from ChromaDB query results."""
    documents = truncate_contexts(results["documents"][0])
    return documents, "\n\n".join(documents)
