from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME
from vector_store import get_retriever
from generation import generate_final_answer

chat_history = []  

def ask_question(user_query):
    if chat_history:
        messages = [
            SystemMessage(content="You are a strict search query generator. Rewrite the user's new question to be standalone based on the chat history. OUTPUT EXACTLY THE REWRITTEN QUESTION ONLY. Do not add 'Here is the question' or any other conversational text.")
        ] + chat_history + [HumanMessage(content=f"New Question: {user_query}")]

        llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        search_query = llm.invoke(messages).content.strip()
    else:
        search_query = user_query
        
    docs = get_retriever().invoke(search_query)
    if len(docs) == 0 and chat_history:
        docs = get_retriever().invoke(user_query)

    # Safely extract and format page numbers
    extracted_pages = []
    for doc in docs:
        page_val = doc.metadata.get("page")
        if page_val is None:
            page_val = doc.metadata.get("page_number")
            
        if page_val is not None:
            extracted_pages.append(int(page_val) + 1)
            
    source_pages = list(set(extracted_pages))
    source_pages.sort()
    
    answer = generate_final_answer(docs, user_query, chat_history)  
    
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer, additional_kwargs={"source_pages": source_pages}))
    
    return answer, source_pages