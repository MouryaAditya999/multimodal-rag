import os
import fitz
import docx


def _record(page_num, text, source, file_format, modality="text"):
    return {
        "page": page_num,
        "text": text,
        "source": os.path.basename(source),
        "format": file_format,
        "modality": modality,
    }


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if text:
            pages.append(_record(page_num + 1, text, pdf_path, "pdf"))
    doc.close()
    return pages


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    pages = []
    current_text = []
    char_count = 0
    page_num = 1
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current_text.append(text)
        char_count += len(text)
        if char_count >= 1500:
            pages.append(
                _record(page_num, "\n".join(current_text), docx_path, "docx")
            )
            page_num += 1
            current_text = []
            char_count = 0
    if current_text:
        pages.append(
            _record(page_num, "\n".join(current_text), docx_path, "docx")
        )
    return pages


def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    pages = []
    chunk_size = 1500
    for i in range(0, len(content), chunk_size):
        pages.append(
            _record(
                (i // chunk_size) + 1,
                content[i : i + chunk_size],
                txt_path,
                "txt",
            )
        )
    return pages


def extract_tables_from_content(file_path):
    from app.ingestion.table_extractor import extract_tables as _extract_tables
    return _extract_tables(file_path)


def extract_images_from_content(file_path, images_dir, use_blip=False, use_clip=False):
    from app.ingestion.image_extractor import (
        extract_images_from_pdf,
        extract_image_from_file,
        generate_image_captions,
        generate_clip_embeddings,
    )

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        images = extract_images_from_pdf(file_path, images_dir)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        images = extract_image_from_file(file_path)
    else:
        return []

    if images:
        images = generate_image_captions(images, use_blip=use_blip)
        for img in images:
            img["text"] = img.get("caption", "")
        if use_clip:
            images = generate_clip_embeddings(images)
    return images


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        pages = extract_text_from_docx(file_path)
    elif ext in [".txt", ".md"]:
        pages = extract_text_from_txt(file_path)
    else:
        return []

    return pages


def load_multimodal_content(file_path):
    content = {"texts": [], "tables": [], "images": []}
    ext = os.path.splitext(file_path)[1].lower()

    from app.config import BASE_DIR, EXTRACT_TABLES, EXTRACT_IMAGES, USE_BLIP_CAPTIONING, USE_CLIP_EMBEDDINGS

    images_dir = os.path.join(BASE_DIR, "data", "images")
    os.makedirs(images_dir, exist_ok=True)

    # Text extraction
    if ext in (".pdf", ".docx", ".txt", ".md"):
        if ext == ".pdf":
            content["texts"] = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            content["texts"] = extract_text_from_docx(file_path)
        elif ext in (".txt", ".md"):
            content["texts"] = extract_text_from_txt(file_path)

    # Table extraction
    if EXTRACT_TABLES and ext in (".pdf", ".csv", ".xlsx"):
        content["tables"] = extract_tables_from_content(file_path)

    # Image extraction
    if EXTRACT_IMAGES and (ext == ".pdf" or ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        content["images"] = extract_images_from_content(file_path, images_dir, use_blip=USE_BLIP_CAPTIONING, use_clip=USE_CLIP_EMBEDDINGS)

    return content
