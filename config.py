from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "db/chromadb"   
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
VISION_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 5
RETRIEVER_SCORE_THRESHOLD = 0.3