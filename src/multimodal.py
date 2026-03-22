import os
import uuid
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import fitz
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, TableTransformerForObjectDetection

from constants import (
    ASSETS_DIR,
    MAX_TABLES_PER_PAGE,
    PDF_RENDER_SCALE,
    QDRANT_URL,
    TABLE_DETECTION_MODEL_NAME,
    TABLE_DETECTION_THRESHOLD,
    VISUAL_COLLECTION_NAME,
    VISUAL_EMBED_MODEL_NAME,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_visual_encoder():
    processor = AutoProcessor.from_pretrained(VISUAL_EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(VISUAL_EMBED_MODEL_NAME)
    model.eval()
    model.to(get_device())
    return processor, model


@lru_cache(maxsize=1)
def get_table_detector():
    processor = AutoImageProcessor.from_pretrained(TABLE_DETECTION_MODEL_NAME)
    model = TableTransformerForObjectDetection.from_pretrained(TABLE_DETECTION_MODEL_NAME)
    model.eval()
    model.to(get_device())
    return processor, model


@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def normalize_vector(vector: torch.Tensor) -> List[float]:
    vector = vector.detach().cpu().float()
    vector = vector / (vector.norm(p=2) + 1e-12)
    return vector.tolist()


def embed_text_for_visual_search(text: str) -> List[float]:
    processor, model = get_visual_encoder()
    device = get_device()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return normalize_vector(features[0])


def embed_image_file(image_path: str) -> List[float]:
    processor, model = get_visual_encoder()
    device = get_device()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return normalize_vector(features[0])


def qdrant_visual_query(query_vector: List[float], limit: int, source_filter: Optional[List[str]] = None):
    client = get_qdrant_client()
    query_filter = None
    if source_filter:
        query_filter = rest.Filter(
            must=[rest.FieldCondition(key="source", match=rest.MatchAny(any=source_filter))]
        )

    try:
        response = client.query_points(
            collection_name=VISUAL_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        points = getattr(response, "points", response)
    except Exception:
        points = client.search(
            collection_name=VISUAL_COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

    results = []
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        score = getattr(point, "score", 0.0)
        results.append({"payload": payload, "score": score})
    return results


def ensure_asset_dir_for_source(source_path: str) -> str:
    safe_name = os.path.basename(source_path).replace(".", "_")
    target_dir = os.path.join(ASSETS_DIR, safe_name)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def render_pdf_pages_to_images(pdf_path: str) -> List[Dict]:
    asset_dir = ensure_asset_dir_for_source(pdf_path)
    doc = fitz.open(pdf_path)
    results = []
    try:
        matrix = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = os.path.join(asset_dir, f"page_{page_index + 1}.png")
            pix.save(image_path)
            page_text = page.get_text("text") or ""
            results.append(
                {
                    "page_number": page_index + 1,
                    "image_path": image_path,
                    "page_text": page_text,
                }
            )
    finally:
        doc.close()
    return results


def detect_tables_on_page_image(image_path: str) -> List[Tuple[int, int, int, int]]:
    image = Image.open(image_path).convert("RGB")
    processor, model = get_table_detector()
    device = get_device()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs,
        threshold=TABLE_DETECTION_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    boxes = []
    for box in results["boxes"][:MAX_TABLES_PER_PAGE]:
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        boxes.append((x0, y0, x1, y1))
    return boxes


def extract_pdf_text_from_bbox(pdf_path: str, page_number: int, bbox: Tuple[int, int, int, int]) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        scale = PDF_RENDER_SCALE
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale)
        return page.get_text("text", clip=rect) or ""
    finally:
        doc.close()


def crop_table_image(image_path: str, bbox: Tuple[int, int, int, int], crop_name: str) -> str:
    image = Image.open(image_path).convert("RGB")
    cropped = image.crop(bbox)
    crop_path = os.path.join(os.path.dirname(image_path), crop_name)
    cropped.save(crop_path)
    return crop_path


def build_visual_assets_for_file(file_path: str) -> List[Dict]:
    lower = file_path.lower()
    assets: List[Dict] = []

    if lower.endswith(".pdf"):
        page_images = render_pdf_pages_to_images(file_path)
        for page_info in page_images:
            page_number = page_info["page_number"]
            image_path = page_info["image_path"]
            page_text = (page_info["page_text"] or "").strip()

            assets.append(
                {
                    "asset_id": f"{file_path}::page::{page_number}",
                    "source": file_path,
                    "asset_path": image_path,
                    "asset_type": "page_image",
                    "page": page_number,
                    "bbox": None,
                    "description": f"Rendered PDF page {page_number} from {os.path.basename(file_path)}",
                    "text_context": page_text[:3000],
                }
            )

            table_boxes = detect_tables_on_page_image(image_path)
            for table_idx, bbox in enumerate(table_boxes, start=1):
                crop_path = crop_table_image(image_path, bbox, f"page_{page_number}_table_{table_idx}.png")
                table_text = extract_pdf_text_from_bbox(file_path, page_number, bbox).strip()
                assets.append(
                    {
                        "asset_id": f"{file_path}::table::{page_number}::{table_idx}",
                        "source": file_path,
                        "asset_path": crop_path,
                        "asset_type": "table_crop",
                        "page": page_number,
                        "bbox": ",".join(str(v) for v in bbox),
                        "description": f"Detected table {table_idx} on page {page_number} from {os.path.basename(file_path)}",
                        "text_context": table_text[:3000],
                    }
                )
        return assets

    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        assets.append(
            {
                "asset_id": f"{file_path}::image::1",
                "source": file_path,
                "asset_path": file_path,
                "asset_type": "image_file",
                "page": None,
                "bbox": None,
                "description": f"Standalone image asset {os.path.basename(file_path)}",
                "text_context": os.path.basename(file_path),
            }
        )

    return assets


def delete_asset_directory_for_source(source_path: str) -> None:
    source_dir = ensure_asset_dir_for_source(source_path)
    if not os.path.exists(source_dir):
        return

    for root, dirs, files in os.walk(source_dir, topdown=False):
        for file_name in files:
            os.remove(os.path.join(root, file_name))
        for dir_name in dirs:
            os.rmdir(os.path.join(root, dir_name))

    try:
        os.rmdir(source_dir)
    except OSError:
        pass


def upsert_visual_assets(assets: List[Dict]) -> None:
    if not assets:
        return

    client = get_qdrant_client()
    points = []
    for asset in assets:
        vector = embed_image_file(asset["asset_path"])
        payload = {
            "asset_id": asset["asset_id"],
            "source": asset["source"],
            "asset_path": asset["asset_path"],
            "asset_type": asset["asset_type"],
            "page": asset["page"],
            "bbox": asset["bbox"],
            "description": asset["description"],
            "text_context": asset["text_context"],
        }
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, asset["asset_id"]))
        points.append(rest.PointStruct(id=qdrant_id, vector=vector, payload=payload))

    client.upsert(collection_name=VISUAL_COLLECTION_NAME, points=points)


def visual_search(query: str, selected_sources: Optional[List[str]], limit: int) -> List[Dict]:
    query_vector = embed_text_for_visual_search(query)
    matches = qdrant_visual_query(query_vector, limit=limit, source_filter=selected_sources)

    results = []
    for match in matches:
        payload = match.get("payload", {})
        results.append(
            {
                "chunk_id": payload.get("asset_id"),
                "source": payload.get("source", "Unknown"),
                "title": payload.get("asset_type", "visual_asset"),
                "section_path": "",
                "page": payload.get("page"),
                "modality": "visual",
                "summary": payload.get("description", ""),
                "text": (
                    f"Visual Asset Type: {payload.get('asset_type', 'unknown')}\n"
                    f"Description: {payload.get('description', '')}\n"
                    f"Associated Text: {payload.get('text_context', '')}\n"
                    f"Asset Path: {payload.get('asset_path', '')}"
                ),
                "raw_text": payload.get("text_context", ""),
                "asset_path": payload.get("asset_path"),
                "asset_type": payload.get("asset_type"),
                "retrieval_mode": "visual",
                "score": match.get("score", 0.0),
            }
        )
    return results