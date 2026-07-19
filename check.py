from vector_store import get_vector_store
from config import CHROMA_DIR

print("CHROMA_DIR:", CHROMA_DIR)
db = get_vector_store()
data = db.get()
print("Document count:", len(data["documents"]))