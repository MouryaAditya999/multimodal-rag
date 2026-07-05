import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Multimodal RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid_rerank"
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    contexts: list
    latencies: dict


class IngestResponse(BaseModel):
    status: str
    message: str


# Lazy-loaded references
_embedder = None
_llm = None
_retriever = None


def _get_models():
    global _embedder, _llm, _retriever
    if _embedder is None:
        from app.embeddings.embedder import embed_text
        _embedder = embed_text
    if _llm is None:
        from app.llm.local_llm import llm as mistral_llm, generate_answer
        _llm = generate_answer
    if _retriever is None:
        from app.vectorstore.retriever import get_context_for_query
        _retriever = get_context_for_query
    return _embedder, _llm, _retriever


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    from app.utils import truncate_contexts

    _, generate_answer_fn, retrieve_fn = _get_models()

    t0 = time.time()

    if request.mode == "dense":
        results = retrieve_fn(request.question, mode="dense")
        documents = truncate_contexts(results["documents"][0])
        metadatas = results["metadatas"][0][: len(documents)]
    else:
        results = retrieve_fn(request.question, mode=request.mode)
        documents = truncate_contexts(results["documents"])
        metadatas = results["metadatas"][: len(documents)]

    t_retrieval = time.time()
    context = "\n\n".join(documents)
    answer = generate_answer_fn(context, request.question)

    t_end = time.time()

    contexts_out = [
        {
            "text": doc[:300],
            "source": meta.get("source", "?"),
            "page": meta.get("page", "?"),
            "modality": meta.get("modality", "text"),
        }
        for doc, meta in zip(documents, metadatas)
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        mode=request.mode,
        contexts=contexts_out,
        latencies={
            "retrieval_sec": round(t_retrieval - t0, 2),
            "generation_sec": round(t_end - t_retrieval, 2),
            "total_sec": round(t_end - t0, 2),
        },
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    from app.utils import truncate_contexts

    _, generate_answer_fn, retrieve_fn = _get_models()

    async def event_generator():
        t0 = time.time()
        yield {"event": "status", "data": "retrieving"}

        if request.mode == "dense":
            results = retrieve_fn(request.question, mode="dense")
            documents = truncate_contexts(results["documents"][0])
            metadatas = results["metadatas"][0][: len(documents)]
        else:
            results = retrieve_fn(request.question, mode=request.mode)
            documents = truncate_contexts(results["documents"])
            metadatas = results["metadatas"][: len(documents)]

        contexts_out = [
            {
                "text": doc[:300],
                "source": meta.get("source", "?"),
                "page": meta.get("page", "?"),
                "modality": meta.get("modality", "text"),
            }
            for doc, meta in zip(documents, metadatas)
        ]

        yield {
            "event": "contexts",
            "data": contexts_out,
        }

        context = "\n\n".join(documents)

        yield {"event": "status", "data": "generating"}

        answer = generate_answer_fn(context, request.question)
        t_end = time.time()

        yield {
            "event": "answer",
            "data": {
                "answer": answer,
                "latencies": {
                    "total_sec": round(t_end - t0, 2),
                },
            },
        }

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.post("/ingest", response_model=IngestResponse)
async def run_ingestion():
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "main.py", "ingest"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return IngestResponse(status="success", message="Ingestion completed.")
        else:
            return IngestResponse(
                status="error", message=result.stderr[-500:]
            )
    except Exception as e:
        return IngestResponse(status="error", message=str(e))
