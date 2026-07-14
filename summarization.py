import json
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from config import LLM_MODEL_NAME, VISION_MODEL_NAME
from ingestion import seperate_content_types

def create_ai_summary(text: str, tables: List[str], images: List[str]):


    try:
        llm = ChatGroq(
            model=LLM_MODEL_NAME,
            temperature=0
        )

        prompt_text = f"""You are creating a searchable description for a document content retrieval. 
        
        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}
        
        """
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}: \n{table}\n\n"

        prompt_text += """
                YOUR TASK: 
                Generate a comprehensive, searchable description that covers:
                
                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION: """

        if  images:
            llm = ChatGroq(model=VISION_MODEL_NAME, temperature=0)
            message_content = [{"type": "text", "text": prompt_text}]

            for image_base64 in images:
                message_content.append({
                    "type": "image_url",
                    "image url": {"url" : f"data:image/jpeg;base64,{image_base64}"}

                })

            message = HumanMessage(content = message_content)
        else: 
            llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
            message = HumanMessage(content = prompt_text)

        response = llm.invoke([message])
        return response.content
    
    except Exception as e: 
        print(f" AI Summary unavailable : {e}")
    
        summary = f"{text[:300]}..."

        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"

        return summary


def summarize_chunks(chunks):
    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f" Processing chunk {current_chunk}/{total_chunks}")

        content_data = seperate_content_types(chunk)

        print(f" Types found: {content_data['types']}")
        print(f" Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        if content_data['tables'] or content_data['images']:
            print("Creating image/table summaries....")
            try:
                enhanced_content = create_ai_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
                print("summary created successfully")

            except Exception as e:
                print(f" AI Summary unavailable : {e}")
                enhanced_content = content_data['text']
        else:
            print(f" Using raw text (no tables/images in content)")
            enhanced_content = content_data['text']

        doc = Document(
            page_content = enhanced_content,
            metadata = {
                "original_content" : json.dumps({
                    "raw_text" : content_data['text'],
                    "tables_html" : content_data['tables'],
                    "images_base64":  content_data['images']
                })

            }
        )
        langchain_documents.append(doc)
    
    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents