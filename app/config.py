import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

LLM_MODEL_PATH = os.path.join(MODELS_DIR, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Chunking (per project spec)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 8
TOP_K_DENSE = 20
TOP_K_BM25 = 20
TOP_K_RERANK = 5
TOP_K_HYBRID = 20

# RRF weights (per project spec)
RRF_WEIGHT_DENSE = 0.6
RRF_WEIGHT_BM25 = 0.4

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

# Reranker (optional - improves precision)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

# Multimodal options
EXTRACT_TABLES = True
EXTRACT_IMAGES = True
USE_BLIP_CAPTIONING = False       # requires ~935MB model download
USE_CLIP_EMBEDDINGS = False       # requires ~338MB model download

# LLM settings
LLM_N_CTX = 2048
LLM_N_THREADS = 8
LLM_N_GPU_LAYERS = 0
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.2
