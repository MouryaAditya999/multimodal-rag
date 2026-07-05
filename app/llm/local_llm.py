from llama_cpp import Llama

from app.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
)

print("Loading Mistral 7B...")

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_N_CTX,
    n_threads=LLM_N_THREADS,
    n_gpu_layers=LLM_N_GPU_LAYERS,
    verbose=False,
)

print("Mistral Loaded Successfully!")


def generate_answer(context, question, max_context_chars=2500):
    """Generate answer, truncating context if it exceeds the model window."""
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...[context truncated]"

    prompt = f"""[INST]
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}
[/INST]
"""

    output = llm(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=0.95,
        stop=["</s>"],
    )

    return output["choices"][0]["text"].strip()
