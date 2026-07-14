import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME, VISION_MODEL_NAME


def generate_final_answer(chunks, query, chat_history=None):
    chat_history = chat_history or []
    try:
        all_images = []

        prompt_text = f"""Based on the following documents, please answer this question: {query}

    CONTENT TO ANALYZE:
    """
        for i, chunk in enumerate(chunks):
            prompt_text += f"Document {i}\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"

                all_images.extend(original_data.get("images_base64", []))

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        if all_images:
            llm = ChatGroq(model=VISION_MODEL_NAME, temperature=0)

            message_content = [{"type": "text", "text": prompt_text}]
            for image_base64 in all_images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            human_msg = HumanMessage(content=message_content)
        else:
            llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
            human_msg = HumanMessage(content=prompt_text)  # plain string, no image blocks

        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history.")
        ] + chat_history + [human_msg]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."