import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import TextLoader, DirectoryLoader #to load data from files
from langchain_text_splitters import CharacterTextSplitter #to create chunks from the data 
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(path = docs_path, glob="*.txt", loader_cls = TextLoader, loader_kwargs={"encoding": "utf-8"},)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in  {docs_path}")
    
    # for i, doc in enumerate (documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f" Source: {doc.metadata['source']}")
    #     print(f" Content Length: {len(doc.page_content)} characters")
    #     print(f" Content preview: {doc.page_content[:100]}...")
    #     print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size = 800, chunk_overlap = 0):
    print("Splitting documents into chunks ")
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f" Source: {chunk.metadata['source']}")
    #         print(f" Content Length: {len(chunk.page_content)} characters")
    #         print(f" Content:")
    #         print(chunk.page_content)
    #         print("-"*50)

    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks
        

def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space" : "cosine"} #cosine similarity algorithm 
    )
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main Function")
    # Creating the ingestion pipeline 
    #1. Loading the source document 
    documents = load_documents(docs_path="docs")

    #2. Chunking 
    chunks = split_documents(documents)

    #3. Embedding 
    #4. Storing vector embeddings in ChromaDB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()

