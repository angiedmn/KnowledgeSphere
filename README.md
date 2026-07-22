# KnowledgeSphere: AI-Powered RAG Knowledge Assistant

KnowledgeSphere is a document Q&A assistant built on a Retrieval-Augmented Generation (RAG) pipeline. Upload a PDF, and KnowledgeSphere parses its text, tables, and images, indexes it for hybrid search, and lets you ask natural-language questions — with every answer citing the exact source page, and a clickable in-app PDF viewer that jumps straight to it.

---
<span style="color:red">**⚠️ WARNING: We advise users to avoid uploading highly sensitive or confidential information unless they're set up with a secure enterprise agreement.**</span>

## ✨ Features

- **Multi-format PDF ingestion** — extracts text, tables (with structure preserved), and images from PDFs using high-resolution layout analysis and OCR.
- **AI-enhanced summarization** — each document chunk is enriched with an LLM-generated searchable description before embedding, improving retrieval quality beyond raw text matching.
- **Hybrid retrieval** — combines dense vector search (Chroma + sentence-transformers embeddings) with sparse keyword search (BM25), fused via Reciprocal Rank Fusion (RRF).
- **Multi-query expansion** — automatically rephrases each question into multiple variations to retrieve a broader, more relevant set of chunks.
- **Conversational memory** — follow-up questions are automatically rewritten into standalone queries using prior chat history.
- **Page-level citations** — every claim in an answer is tagged with the exact PDF page it came from, in the format `[Page X]`.
- **Clickable source viewer** — click a cited page number to jump the embedded PDF viewer directly to that page, similar to Claude/Gemini's document citation experience.
- **Vision-aware answers** — automatically routes to a vision-capable model when a question is about images, figures, or diagrams, and pulls in the relevant image content.
- **Re-ingestion safe** — re-uploading the same file automatically removes its old chunks first, so the knowledge base never accumulates duplicates.
- **Streamlit web UI** — a full chat interface with document upload, animated ingestion progress, and a source citation panel, plus a simple CLI (`app.py`) for quick local testing.


## 🧰 Tech Stack

| Layer                | Technology                                                        |
|-----------------------|--------------------------------------------------------------------|
| UI                    | Streamlit                                                          |
| LLM Orchestration     | LangChain, LangChain-Groq                                          |
| LLM Provider          | Groq (Llama 3.3 70B, Llama 4 Scout/Maverick for vision)             |
| Document Parsing      | `unstructured` (hi-res PDF partitioning), Tesseract OCR             |
| Vector Store          | ChromaDB                                                            |
| Embeddings            | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)              |
| Keyword Search        | BM25 (`rank_bm25`)                                                   |
| Language              | Python 3.13                                                         |

---

## 📁 Project Structure

```
RAG/
├── app.py                     # CLI entry point (ingest + REPL Q&A loop)
├── streamlit_app.py           # Web UI — chat interface, upload, citations, PDF viewer
├── config.py                  # Model names, retriever settings, paths
├── ingestion.py               # PDF partitioning, chunking, content separation
├── summarization.py           # AI summarization of table/image chunks (Groq text/vision)
├── vector_store.py            # Chroma + BM25 setup, add/delete/retrieve
├── multi_query.py             # Query expansion, hybrid retrieval, RRF fusion
├── conversational_RAG.py      # Question rewriting, retrieval orchestration, chat history
├── generation.py              # Final answer generation with page citations
├── retrieval_profiles.py      # translates UI sliders into retrieval
├── requirements.txt
└── .env                            # API keys (not committed)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/keys)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally (required by `unstructured` for scanned/image-heavy PDFs)

### Installation

```bash
#1 Clone the repository: 
git clone https://github.com/<your-username>/RAG.git
cd KnowledgeSphere

#2 Create and activate a virtual environment:  
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source venv/bin/activate

#3 Install dependencies:  
pip install -r requirements.txt
```

### Configure environment variables

#4 Create a `.env` file and add the required API keys: 
```
GROQ_API_KEY=your_groq_api_key_here
```

#5 Local installation of Tesseract OCR (https://tesseract-ocr.github.io/tessdoc/Installation.html)
(Installation of Poppler: https://poppler.freedesktop.org/)

### Run the web app

#6 Launch the application:
```bash
streamlit run streamlit_app.py
```

### Run the CLI version

```bash
python app.py
```

---

## 🧪 Usage

1. Launch the Streamlit application in a web browser.
2. Navigate to the sidebar under Manage Knowledge Base and upload a PDF 
3. Click Process Document and wait for the ingestion pipeline to extract text, tables, and images and build the vector database.
4. Adjust the Search Focus slider to prioritize semantic understanding or keyword matching, depending on the retrieval requirements and the Answer Style slider to choose between concise, direct responses and more detailed, exploratory explanations.
5. Enter a question related to the uploaded document in the chat interface.
6. The generated response is provided along with citations. Click any page number button beneath the generated answer to view the source text in the embedded PDF viewer.
7. Continue asking follow-up questions. The system maintains conversation history to support context-aware interactions without requiring the user to restate previous queries.
8. Click on ‘Clear Database’ to start afresh with new documents.


---

## ⚠️ Known Limitations

- OCR quality depends on Tesseract and the source PDF's scan quality.
- Groq's free/on-demand tier has daily token limits — heavy ingestion of large or table/image-dense PDFs can exhaust the daily quota quickly.
- Citation accuracy depends on the LLM correctly following the `[Page X]` instruction; occasional missed or imprecise citations are possible.
- Chat history is currently stored as global in-process state rather than per-session — intended for single-user local use, not concurrent multi-user deployment.

---

## 👥 Team

| **RAG Pipeline & Backend Engineer** | Document ingestion, chunking, embedding pipeline, environment/config setup, testing, query fusion, LLM prompt/answer generation, response style functioning, RAGAS Evaluation | Aashna Sharma |

| **Frontend & UX Engineer**          | Streamlit chat interface, PDF viewer integration, citation UI, citation logic, ingestion progress experience, styling, RAGAS testing | Angel Dhiman |

| **Data & Infrastructure Engineer**  | Vector store management, hybrid retrieval (vector + BM25), RRF Fusion, documentation | Hiya Ratra |


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to your fork (`git push origin feature/your-feature`)
5. Open a Pull Request

### Debugging notes

If retrieval seems to be returning content from documents other than the one currently loaded, or repeated chunks, this usually means the vector store wasn't cleared before a new ingestion. Use "Clear Database" in the sidebar, or call clear_all_documents() from vector_store.py, before re-testing.

If a chunk's embedding seems to under-represent a specific term despite it being in the source text, check chunking granularity first (CHUNK_MAX_CHARACTERS / CHUNK_COMBINE_UNDER_N_CHARS) before assuming a retrieval-threshold or summarization-prompt issue — overly large chunks that merge multiple concepts are the most common cause.
