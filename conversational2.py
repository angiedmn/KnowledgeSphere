from mul_query import multi_query_hybrid_retrieve
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME
from vector_store import get_retriever
from generation import generate_final_answer
from Hybrid_retriever import get_bm25_retriever

chat_history = []  

def ask_question(user_query):
    # 1. Reformulate the question based on chat history
    if chat_history:
        messages = [
            SystemMessage(content="You are a strict search query generator. Rewrite the user's new question to be standalone based on the chat history. OUTPUT EXACTLY THE REWRITTEN QUESTION ONLY. Do not add 'Here is the question' or any other conversational text.")
        ] + chat_history + [HumanMessage(content=f"New Question: {user_query}")]

        llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        search_query = llm.invoke(messages).content.strip()
    else:
        search_query = user_query
        
   # 2. Retrieve documents using Multi-Query + Hybrid (Vector+BM25) + Reciprocal Rank Fusion
    retriever = get_retriever()
    bm25_retriever = get_bm25_retriever()
    docs = multi_query_hybrid_retrieve(search_query, retriever, bm25_retriever)
    
    # 3. Fallback: If no docs found, try again with the raw user query
    if len(docs) == 0 and chat_history:
        docs = multi_query_hybrid_retrieve(user_query, retriever, bm25_retriever)
    
    print(f"Found {len(docs)} relevant documents (after multi-query + RRF)")
    
    # 4. Safely extract and format page numbers
    extracted_pages = []
    for doc in docs:
        page_val = doc.metadata.get("page")
        if page_val is None:
            page_val = doc.metadata.get("page_number")
            
        if page_val is not None:
            extracted_pages.append(int(page_val) + 1)
            
    source_pages = list(set(extracted_pages))
    source_pages.sort()
    
    # 5. Generate the final answer using the fused documents
    answer = generate_final_answer(docs, user_query, chat_history)  
    
    # 6. Update chat history
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer, additional_kwargs={"source_pages": source_pages}))
    
    return answer, source_pages

# ==========================================
# CONVERSATIONAL TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" Conversational RAG Terminal Test")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting chat test...")
            break
            
        try:
            # Call your main function
            test_answer, test_source_pages = ask_question(user_input)
            
            print(f"\nAI Answer: {test_answer}")
            print(f"Source Pages: {test_source_pages}")
            
            # This prints exactly how many messages the AI is currently remembering!
            print(f"--- [Debug: Chat History contains {len(chat_history)} messages] --- \n")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
            