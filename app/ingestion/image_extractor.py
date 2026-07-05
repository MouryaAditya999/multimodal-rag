import os
import hashlib
from PIL import Image
from io import BytesIO


IMAGES_DIR = None


def _get_images_dir(base_dir):
    global IMAGES_DIR
    if IMAGES_DIR is None:
        IMAGES_DIR = os.path.join(base_dir, "data", "images")
        os.makedirs(IMAGES_DIR, exist_ok=True)
    return IMAGES_DIR


def extract_images_from_pdf(pdf_path, output_dir):
    import fitz

    images_data = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            img_hash = hashlib.md5(image_bytes).hexdigest()[:12]
            filename = f"{os.path.basename(pdf_path)}_p{page_num+1}_{img_idx}_{img_hash}.{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            images_data.append(
                {
                    "page": page_num + 1,
                    "source": os.path.basename(pdf_path),
                    "format": "pdf",
                    "modality": "image",
                    "image_path": filepath,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0),
                }
            )
    doc.close()
    return images_data


def extract_image_from_file(file_path):
    img = Image.open(file_path)
    w, h = img.size
    return [
        {
            "page": 1,
            "source": os.path.basename(file_path),
            "format": os.path.splitext(file_path)[1][1:],
            "modality": "image",
            "image_path": file_path,
            "width": w,
            "height": h,
        }
    ]


def generate_image_captions(images_data, use_blip=False):
    """Generate captions for images.
    If use_blip=True, uses BLIP model (requires transformers).
    Otherwise, generates a simple descriptive caption from metadata.
    """
    if use_blip:
        try:
            return _generate_blip_captions(images_data)
        except Exception as e:
            print(f"  BLIP captioning failed ({e}), using fallback.")

    for img in images_data:
        fname = os.path.basename(img["image_path"])
        w, h = img.get("width", 0), img.get("height", 0)
        img["caption"] = f"Image from {img['source']}, page {img.get('page', 1)} ({fname}, {w}x{h}px)"
    return images_data


def _generate_blip_captions(images_data):
    from transformers import BlipProcessor, BlipForConditionalGeneration

    print("Loading BLIP model for image captioning...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    print("BLIP loaded.")

    for img_data in images_data:
        try:
            image = Image.open(img_data["image_path"]).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            img_data["caption"] = caption
            print(f"  Caption: {caption}")
        except Exception as e:
            img_data["caption"] = f"Image from {img_data['source']}, page {img_data.get('page', 1)}"
            print(f"  Caption failed: {e}")

    return images_data


def generate_clip_embeddings(images_data):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("clip-ViT-B-32")
    print("CLIP model loaded for image embeddings.")

    for img_data in images_data:
        try:
            image = Image.open(img_data["image_path"]).convert("RGB")
            embedding = model.encode(image, convert_to_numpy=True)
            img_data["embedding"] = embedding.tolist()
        except Exception as e:
            img_data["embedding"] = None
            print(f"  CLIP embedding failed: {e}")

    return images_data
