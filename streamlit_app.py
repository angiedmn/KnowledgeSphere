import streamlit as st
import os
import re
from dotenv import load_dotenv
load_dotenv()
import shutil
from langchain_core.messages import HumanMessage, AIMessage
from ingestion import partition_document, create_chunks_by_title
from summarization import summarize_chunks
from vector_store import add_documents, get_vector_store, delete_by_source
from conversational_RAG import ask_question
import config
import conversational_RAG
import base64
import random
import time
import threading


#PAGE CONFIG

st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

#load css

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

#int session state

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'viewer_page' not in st.session_state:
    st.session_state.viewer_page = None

if 'vector_store_status' not in st.session_state:
    try:
        db = get_vector_store()
        st.session_state.vector_store_status = "Database loaded"
    except Exception:
        st.session_state.vector_store_status = "No database found"

TEMP_UPLOAD_DIR = "temp_ingest"

#helper fns

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        overlay = st.empty()
        render_loading_overlay(overlay, 0, "Starting...")

        from vector_store import delete_by_source
        delete_by_source(file_path)

        elements = run_with_animated_progress(
            overlay, lambda: partition_document(file_path), 5, 20, "Partitioning PDF..."
        )
        chunks = run_with_animated_progress(
            overlay, lambda: create_chunks_by_title(elements), 20, 28, "Creating chunks..."
        )
        documents = run_summarize_with_progress(overlay, chunks, file_path, 28, 90)
        run_with_animated_progress(
            overlay, lambda: add_documents(documents, source_file=file_path), 90, 99, "Embedding & storing..."
        )

        render_loading_overlay(overlay, 100, "Done!")
        time.sleep(0.4)
        overlay.empty()

        st.session_state.current_pdf_path = file_path
        st.session_state.vector_store_status = "Database loaded"
        st.session_state.chat_history = []
        st.rerun()

def display_pdf(file_path, page_num):
    """Embeds the PDF in an iframe and jumps to the specific page."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Source document is no longer available.")

def start_new_chat():
    st.session_state.chat_history = []
    st.rerun()

def extract_cited_pages(answer_text: str, fallback_pages):
    """Pull every page number the model actually cited via [Page X] tags.
    Falls back to the retriever's page list if the model didn't cite anything."""
    cited = sorted({int(n) for n in re.findall(r"\[Page\s+(\d+)\]", answer_text)})
    return cited if cited else list(fallback_pages)

def render_sources(msg_key: str, answer_text: str, fallback_pages):
    """Render one clickable button per cited page. Clicking jumps the PDF
    viewer straight to that page, the way Claude/Gemini do it."""
    cited_pages = extract_cited_pages(answer_text, fallback_pages)
    if not cited_pages or 'current_pdf_path' not in st.session_state:
        return
    st.caption("Sources")
    cols = st.columns(min(len(cited_pages), 6))
    for i, page in enumerate(cited_pages):
        with cols[i % len(cols)]:
            if st.button(f"📄 Page {page}", key=f"{msg_key}_page_{page}", use_container_width=True):
                st.session_state.viewer_page = page

    # If one of this message's pages is the one currently selected, show the
    # viewer right under this message so it feels attached to the citation.
    if st.session_state.viewer_page in cited_pages:
        with st.expander(f"📄 Viewing Page {st.session_state.viewer_page}", expanded=True):
            display_pdf(st.session_state.current_pdf_path, st.session_state.viewer_page)

NUM_SEGMENTS = 16

def render_loading_overlay(placeholder, percent: int, label: str):
    # ease-out curve: fills faster early, slows near the end — feels quicker overall
    eased = 1 - (1 - percent / 100) ** 2
    filled = round(eased * NUM_SEGMENTS)

    segments_html = ""
    for i in range(NUM_SEGMENTS):
        cls = "segment filled" if i < filled else "segment"
        segments_html += f"<div class='{cls}'></div>"

    html = f"""
    <div class='loading-overlay'>
        <div class='loading-title'>Ingesting document</div>
        <div class='loading-subtitle'>{label}</div>
        <div class='segment-bar'>{segments_html}</div>
        <div class='loading-percent'>{percent}%</div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)

FILLER_MESSAGES = [
    "Analyzing document structure...",
    "Extracting semantic patterns...",
    "Cross-referencing content blocks...",
    "Building searchable index...",
    "Optimizing chunk boundaries...",
    "Parsing visual elements...",
    "Linking related concepts...",
    "Compressing knowledge graph...",
    "Refining embeddings...",
    "Indexing key terminology...",
]

def run_with_animated_progress(overlay, work_fn, start_pct, end_pct, base_label, tick_interval=1.3):
    """Runs work_fn() in the background while animating the overlay on a
    fixed, readable cadence in the main thread — decoupled from how long
    the real work actually takes."""
    result_holder = {}
    error_holder = {}

    def _target():
        try:
            result_holder['result'] = work_fn()
        except Exception as e:
            error_holder['error'] = e

    thread = threading.Thread(target=_target)
    thread.start()

    tick = 0
    current_pct = start_pct
    while thread.is_alive():
        label = base_label if tick == 0 else random.choice(FILLER_MESSAGES)
        current_pct = min(current_pct + (end_pct - start_pct) * 0.15, end_pct - 2)
        render_loading_overlay(overlay, round(current_pct), label)
        time.sleep(tick_interval)   # <-- this is what makes ticks evenly spaced & readable
        tick += 1

    thread.join()
    if 'error' in error_holder:
        raise error_holder['error']

    render_loading_overlay(overlay, end_pct, f"{base_label.rstrip('.')} — done")
    time.sleep(0.3)
    return result_holder['result']

def run_summarize_with_progress(overlay, chunks, source_file, start_pct, end_pct, tick_interval=1.3):
    progress_state = {"done": 0, "total": len(chunks)}

    def on_progress(done, total):
        progress_state["done"] = done
        progress_state["total"] = total

    result_holder = {}
    def _target():
        result_holder['result'] = summarize_chunks(chunks, source_file=source_file, progress_callback=on_progress)

    thread = threading.Thread(target=_target)
    thread.start()

    tick = 0
    current_pct = start_pct
    while thread.is_alive():
        done, total = progress_state["done"], progress_state["total"]
        if done > 0 and tick % 2 == 0:
            label = f"Summarizing chunk {done}/{total}..."
            real_pct = start_pct + (end_pct - start_pct) * (done / total)
            current_pct = max(current_pct, real_pct)
        else:
            label = random.choice(FILLER_MESSAGES)
            current_pct = min(current_pct + (end_pct - start_pct) * 0.04, end_pct - 1)
        render_loading_overlay(overlay, round(current_pct), label)
        time.sleep(tick_interval)
        tick += 1

    thread.join()
    render_loading_overlay(overlay, end_pct, "Summarization complete")
    time.sleep(0.3)
    return result_holder['result']


#sidebar

with st.sidebar:
    st.subheader("Documents")
    st.caption(st.session_state.vector_store_status)

    with st.expander("Manage Knowledge Base"):
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], label_visibility="collapsed")
        if st.button("Process Document", use_container_width=True) and uploaded_file:
            process_uploaded_file(uploaded_file)
            
        if st.button("Clear Database", use_container_width=True):
            try:
                shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
                from vector_store import get_bm25_retriever
                get_bm25_retriever(force_rebuild=True)
                st.session_state.vector_store_status = "No database found"
                st.session_state.chat_history = []
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                
    st.divider()
    
    st.subheader("Recent Questions")
    has_history = False
    for msg in reversed(st.session_state.chat_history):
        if isinstance(msg, HumanMessage):
            has_history = True
            st.markdown(f"<div class='history-btn'> {msg.content[:40]}{'...' if len(msg.content) > 40 else ''}</div>", unsafe_allow_html=True)
            st.write("") 
    
    if not has_history:
        st.caption("No recent questions.")

#main chat

if not st.session_state.chat_history:
    st.markdown("<h2 style='text-align: center; color: #fcd34d; margin-top: 15vh;'>How can I help you with your documents today?</h2>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

            pages = message.additional_kwargs.get("source_pages", [])
            render_sources(f"hist_{id(message)}", message.content, pages)


#chat Input

user_query = st.chat_input("Ask a question about the documents...")
if user_query:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                conversational_RAG.chat_history = st.session_state.chat_history
                
                answer, source_pages = ask_question(user_query) 
                
                st.session_state.chat_history = conversational_RAG.chat_history
                st.markdown(answer)

                render_sources(f"new_{len(st.session_state.chat_history)}", answer, source_pages)
            
            except Exception as e:
                st.error(f"Error generating response: {e}")