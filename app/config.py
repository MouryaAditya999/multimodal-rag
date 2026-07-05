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
TOP_K_RERANK = 5          # Reranker is mandatory per spec: reduces 20 -> 5 best
TOP_K_HYBRID = 20

# RRF weights (per project spec: Dense=0.6, BM25=0.4)
RRF_WEIGHT_DENSE = 0.6
RRF_WEIGHT_BM25 = 0.4

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

# Reranker (mandatory per project spec - largest accuracy improvement)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

# Multimodal options
EXTRACT_TABLES = True
EXTRACT_IMAGES = True
USE_BLIP_CAPTIONING = False       # ~935MB model, disable if low on RAM
USE_CLIP_EMBEDDINGS = True        # ~338MB, enables image similarity search

# OCR for scanned PDFs (optional - install Tesseract or EasyOCR)
USE_OCR = False
OCR_ENGINE = "easyocr"            # "tesseract" or "easyocr"

# Document parsing backend (pymupdf is default, docling is optional)
USE_DOCLING = False               # ~500MB model for advanced PDF layout analysis

# LLM settings
LLM_N_CTX = 2048
LLM_N_THREADS = 8
LLM_N_GPU_LAYERS = 0
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.2
