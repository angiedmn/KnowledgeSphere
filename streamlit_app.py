import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import shutil
from langchain_core.messages import HumanMessage, AIMessage
from ingestion import partition_document, create_chunks_by_title
from summarization import summarize_chunks
from vector_store import add_documents, get_vector_store
from conversational_RAG import ask_question
import config
import conversational_RAG
import base64

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
        
        with st.status(f"Ingesting `{uploaded_file.name}`...") as status:
            st.write("Partitioning PDF...")
            elements = partition_document(file_path)
            st.write("Creating chunks...")
            chunks = create_chunks_by_title(elements)
            st.write("Summarizing chunks...")
            documents = summarize_chunks(chunks)
            st.write("Adding to vector store...")
            add_documents(documents)
            status.update(label="Ingestion complete!", state="complete")
    
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
            if pages and 'current_pdf_path' in st.session_state:
                with st.expander(f"📄 View Source (Page {pages[0]})"):
                    display_pdf(st.session_state.current_pdf_path, pages[0])


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
                
                if source_pages and 'current_pdf_path' in st.session_state:
                    with st.expander(f"📄 View Source (Page {source_pages[0]})"):
                        display_pdf(st.session_state.current_pdf_path, source_pages[0])
            
            except Exception as e:
                st.error(f"Error generating response: {e}")