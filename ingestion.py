import pytesseract

import os

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from config import TESSERACT_PATH
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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
        combine_text_under_n_chars = 500,
        include_orig_elements = True
    )

    print(f"created {len(chunks)} chunks")
    return chunks

def seperate_content_types(chunk):
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text'],
        'page_number': None,
    }

    pages_seen = []

    chunk_page = getattr(chunk.metadata, 'page_number', None) if hasattr(chunk, 'metadata') else None
    if chunk_page is not None:
        pages_seen.append(chunk_page)

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            el_page = getattr(element.metadata, 'page_number', None)
            if el_page is not None:
                pages_seen.append(el_page)

            if element_type == "Table":
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            elif element_type == 'Image':
                image = getattr(element.metadata, "image_base64", None)
                if image:
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    if pages_seen:
        content_data['page_number'] = min(pages_seen)

    content_data['types'] = list(set(content_data['types']))
    return content_data

