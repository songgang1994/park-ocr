import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_REGIONS = [
    ("phase1", 4100, 260, 7240, 2190),
    ("phase2", 1650, 1750, 7180, 3820),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiled OCR on split parking regions and annotate recognized numbers."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-image", required=True, help="Output annotated image.")
    parser.add_argument("--output-csv", required=True, help="Output OCR CSV.")
    parser.add_argument(
        "--cache-dir",
        default="park_ocr/.paddlex_cache",
        help="Local cache dir for PaddleX/PaddleOCR models.",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="OCR language.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Minimum OCR confidence to keep.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1600,
        help="Tile size for OCR crops.",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=1200,
        help="Tile stride for OCR crops.",
    )
    parser.add_argument(
        "--tile-scale",
        type=float,
        default=2.0,
        help="Upscale factor applied before OCR.",
    )
    return parser.parse_args()


def configure_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def iter_ocr_results(result_obj):
    if result_obj is None:
        return
    if isinstance(result_obj, list):
        for item in result_obj:
            yield from iter_ocr_results(item)
        return
    if hasattr(result_obj, "res"):
        yield from iter_ocr_results(result_obj.res)
        return
    if isinstance(result_obj, dict):
        boxes = result_obj.get("rec_polys") or result_obj.get("dt_polys") or []
        texts = result_obj.get("rec_texts") or []
        scores = result_obj.get("rec_scores") or []
        for index, box in enumerate(boxes):
            text = texts[index] if index < len(texts) else ""
            score = float(scores[index]) if index < len(scores) else 0.0
            yield box, text, score


def normalize_box(box) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 4:
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        arr = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
    return arr


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def keep_text(text: str, score: float, min_score: float) -> bool:
    if score < min_score:
        return False
    if not text:
        return False
    return bool(re.fullmatch(r"\d{1,4}#?", text) or re.fullmatch(r"#\d{1,4}", text))


def build_positions(start: int, end: int, tile_size: int, stride: int) -> list[int]:
    if end - start <= tile_size:
        return [start]
    positions = list(range(start, end - tile_size + 1, stride))
    last = end - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tiles(region, tile_size: int, stride: int):
    name, left, top, right, bottom = region
    x_positions = build_positions(left, right, tile_size, stride)
    y_positions = build_positions(top, bottom, tile_size, stride)
    for y in y_positions:
        for x in x_positions:
            yield {
                "region": name,
                "left": x,
                "top": y,
                "right": min(x + tile_size, right),
                "bottom": min(y + tile_size, bottom),
            }


def quad_to_aabb(quad: np.ndarray) -> tuple[float, float, float, float]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou(row_a: dict, row_b: dict) -> float:
    ax1, ay1, ax2, ay2 = row_a["bbox"]
    bx1, by1, bx2, by2 = row_b["bbox"]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def deduplicate(rows: list[dict]) -> list[dict]:
    kept = []
    for row in sorted(rows, key=lambda item: (-item["score"], item["text"])):
        duplicate = False
        cx = (row["bbox"][0] + row["bbox"][2]) / 2.0
        cy = (row["bbox"][1] + row["bbox"][3]) / 2.0
        for existing in kept:
            ex = (existing["bbox"][0] + existing["bbox"][2]) / 2.0
            ey = (existing["bbox"][1] + existing["bbox"][3]) / 2.0
            if row["text"] == existing["text"] and iou(row, existing) > 0.20:
                duplicate = True
                break
            if iou(row, existing) > 0.65:
                duplicate = True
                break
            if row["text"] == existing["text"] and abs(cx - ex) <= 12 and abs(cy - ey) <= 12:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0], item["text"]))


def draw_annotations(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for row in rows:
        quad = row["quad"].astype(np.int32)
        cv2.polylines(annotated, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

        x1, y1, x2, _ = row["bbox"]
        label = row["text"]
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tx = max(int(x1), 0)
        ty = max(int(y1) - 6, text_h + 6)
        cv2.rectangle(
            annotated,
            (tx - 1, ty - text_h - baseline - 3),
            (tx + text_w + 3, ty + 2),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (tx + 1, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "text",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "x3",
        "y3",
        "x4",
        "y4",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            quad = row["quad"].astype(int)
            writer.writerow(
                {
                    "region": row["region"],
                    "text": row["text"],
                    "score": f"{row['score']:.4f}",
                    "x1": int(quad[0][0]),
                    "y1": int(quad[0][1]),
                    "x2": int(quad[1][0]),
                    "y2": int(quad[1][1]),
                    "x3": int(quad[2][0]),
                    "y3": int(quad[2][1]),
                    "x4": int(quad[3][0]),
                    "y4": int(quad[3][1]),
                }
            )


def run_tile_ocr(
    ocr,
    image: np.ndarray,
    tile: dict,
    tile_scale: float,
    min_score: float,
) -> list[dict]:
    crop = image[tile["top"] : tile["bottom"], tile["left"] : tile["right"]]
    if tile_scale != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=tile_scale,
            fy=tile_scale,
            interpolation=cv2.INTER_CUBIC,
        )

    result = ocr.predict(crop, text_det_limit_side_len=max(crop.shape[:2]))
    rows = []
    for box, text, score in iter_ocr_results(result):
        compact = clean_text(text)
        if not keep_text(compact, score, min_score):
            continue

        quad = normalize_box(box) / tile_scale
        quad[:, 0] += tile["left"]
        quad[:, 1] += tile["top"]
        bbox = quad_to_aabb(quad)

        rows.append(
            {
                "region": tile["region"],
                "text": compact,
                "score": score,
                "quad": quad,
                "bbox": bbox,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_image = Path(args.output_image)
    output_csv = Path(args.output_csv)
    cache_dir = Path(args.cache_dir)

    configure_env(cache_dir)

    from paddleocr import PaddleOCR

    image = load_image(image_path)
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
        ocr_version="PP-OCRv4",
        device="cpu",
        enable_mkldnn=False,
        enable_hpi=False,
        cpu_threads=4,
    )

    rows = []
    for region in DEFAULT_REGIONS:
        for tile in generate_tiles(region, args.tile_size, args.tile_stride):
            rows.extend(
                run_tile_ocr(
                    ocr=ocr,
                    image=image,
                    tile=tile,
                    tile_scale=args.tile_scale,
                    min_score=args.min_score,
                )
            )

    rows = deduplicate(rows)
    annotated = draw_annotations(image, rows)

    save_image(output_image, annotated)
    save_csv(output_csv, rows)

    print(f"kept={len(rows)}")
    print(output_image)
    print(output_csv)


if __name__ == "__main__":
    main()
