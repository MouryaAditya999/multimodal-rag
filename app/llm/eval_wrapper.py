from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from llama_cpp import Llama

from app.config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_N_THREADS,
)


class LocalMistralLLM(LLM):
    """
    LangChain wrapper for llama-cpp-python running Mistral 7B.
    Used by RAGAS for fully offline evaluation.
    """

    model_path: str = LLM_MODEL_PATH
    n_ctx: int = LLM_N_CTX
    n_threads: int = LLM_N_THREADS
    n_gpu_layers: int = LLM_N_GPU_LAYERS

    _llm: Any = None

    def __init__(self, llm_instance: Any = None, **data: Any):
        super().__init__(**data)
        if llm_instance is not None:
            object.__setattr__(self, "_llm", llm_instance)
        else:
            object.__setattr__(
                self,
                "_llm",
                Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                ),
            )

    @property
    def _llm_type(self) -> str:
        return "local_mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is None:
            stop = ["</s>"]

        output = self._llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=stop,
            **kwargs,
        )
        return output["choices"][0]["text"].strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}
