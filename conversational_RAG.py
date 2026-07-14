from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME
from vector_store import get_retriever
from generation import generate_final_answer

chat_history = []  


def ask_question(user_query):
    if chat_history:
        messages = [
            SystemMessage(content="From the provided chat history, rewrite the new question to be standalone and searchable. Return the rewritten question only.")
        ] + chat_history + [HumanMessage(content=f"New Question: {user_query}")]

        llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        search_query = llm.invoke(messages).content   
    else:
        search_query = user_query

    docs = get_retriever().invoke(search_query)
    print(f"Found {len(docs)} relevant documents")

    answer = generate_final_answer(docs, user_query, chat_history)  

    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer