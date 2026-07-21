import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from config import LLM_MODEL_NAME, VISION_MODEL_NAME


def get_display_page(chunk) -> str:
    page_num = chunk.metadata.get("page_number")
    if page_num is None:
        page_num = chunk.metadata.get("page")
    return str(int(page_num) + 1) if isinstance(page_num, (int, float)) else "unknown"
def generate_final_answer(chunks, query, chat_history=None):
    chat_history = chat_history or []
    try:
        all_images = []

        query_lower = query.lower()
        image_keywords = [
            "image",
            "images",
            "figure",
            "fig",
            "diagram",
            "graph",
            "chart",
            "visual",
            "picture",
            "illustration"
        ]

        need_images = any(
            keyword in query_lower
            for keyword in image_keywords
        )
        need_tables = any(
            keyword in query_lower
            for keyword in [
                "table",
                "tables",
                "compare",
                "comparison",
                "result",
                "results",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "score",
                "performance",
                "statistics",
                "metric",
                "metrics",
                "data"
            ]
        )

        prompt_text = f"""Based on the following documents, please answer this question: {query}

    CONTENT TO ANALYZE:
    """
        for i, chunk in enumerate(chunks):
            page_label = get_display_page(chunk)
            prompt_text += f"--- Document {i} (PDF Page {page_label}) ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                # tables_html = original_data.get("tables_html", [])
                # if tables_html:
                #     prompt_text += "TABLES:\n"
                #     for j, table in enumerate(tables_html):
                #         prompt_text += f"Table {j+1}:\n{table}\n\n"

                if need_tables:
                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        prompt_text += "TABLES:\n"
                        for j, table in enumerate(tables_html):
                            prompt_text += f"Table {j+1}:\n{table}\n\n"

                if need_images:
                    all_images.extend(
                        original_data.get("images_base64", [])
                    )

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, well-structured answer using the text, tables, and images above. Format your response using Markdown:
- Use bullet points or/and numbered lists when presenting multiple facts, steps, or items
- Use **bold** for key terms or important values
- Use short paragraphs for narrative explanation, and switch to lists whenever you're covering more than one distinct point
- Use sub-headings (###) only for longer answers that cover multiple topics
"""

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above.

CITATION RULES (must follow):
- After every sentence or claim that draws on the sources above, add a citation in the exact format [Page X], using the "PDF Page" number given above the relevant Document.
- If a sentence draws on more than one page, cite each one, e.g. "...as shown in the results [Page 5][Page 7]."
- Only ever cite a page number that was explicitly given to you above. Never guess or invent one. There MUST NEVER appear a cited page number that does not exist in the uploaded document.
- If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents." and do not add citations.

ANSWER:"""

        if need_images and all_images:
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
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history. You always cite the source page number in [Page X] format immediately after any claim drawn from the documents.")
        ] + chat_history + [human_msg]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."