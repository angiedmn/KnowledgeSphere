import os, shutil
from ingestion import partition_document, create_chunks_by_title
from summarization import summarize_chunks
from vector_store import add_documents
from conversational_RAG import ask_question


def ingest_document(file_path: str):
    elements = partition_document(file_path)
    chunks = create_chunks_by_title(elements)
    documents = summarize_chunks(chunks)
    add_documents(documents)
    print("Ingestion completed successfully")


def upload_and_ingest(source_path: str, upload_dir="uploads") -> str:
    """Point this at a file a user picked (e.g. from st.file_uploader,
    after writing it to a temp path) to save + ingest it in one call."""
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, os.path.basename(source_path))
    shutil.copy(source_path, dest_path)
    ingest_document(dest_path)
    return dest_path


if __name__ == "__main__":
    ingest_document("./Attention.pdf")  

    print("Ask me questions! Type 'quit' to exit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        ask_question(query)