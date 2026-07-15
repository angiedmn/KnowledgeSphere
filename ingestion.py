import pytesseract
import os
# 1. Force Windows to add Tesseract to the PATH (varies pc to pc depending on tesseract's file path and environment variables)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
def partition_document(file_path: str):

    elements = partition_pdf(
        filename= file_path,
        strategy="hi_res",
        infer_table_structure=True, #Keep table structure
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True #Store images in base64 
    )

    print(f"Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):

    chunks = chunk_by_title(
        elements,
        max_characters = 3000,
        new_after_n_chars = 2400,
        combine_text_under_n_chars = 500
    )

    print(f"created {len(chunks)} chunks")
    return chunks


#changed
def seperate_content_types(chunk):
    content_data = {
        'text' : chunk.text,
        'tables' : [],
        'images' : [],
        'types' : ['text'],
        'page_number': getattr(chunk.metadata, 'page_number', None) if hasattr(chunk, 'metadata') else None,
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if content_data['page_number'] is None:
                content_data['page_number'] = getattr(element.metadata, 'page_number', None)

            if element_type == "Table":
                content_data['types'].append('table')
                table_html = getattr(element.metadata,'text_as_html', element.text)
                content_data['tables'].append(table_html)

            elif element_type =='Image':
                image = getattr(element.metadata, "image_base64", None)
                if image: 
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data