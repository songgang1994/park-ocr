import argparse
import csv
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from keep_parking_slots import keep_slot_lines_and_numbers, load_image, save_image
except ModuleNotFoundError:
    from .keep_parking_slots import keep_slot_lines_and_numbers, load_image, save_image


@dataclass(frozen=True)
class RegionBox:
    region_id: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parking regions, denoise slot plans, and map slot numbers."
    )
    parser.add_argument("--image", required=True, help="Input parking plan image.")
    parser.add_argument(
        "--output-dir",
        default="park_ocr/output",
        help="Directory used for generated images and CSV files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="park_ocr/.paddlex_cache",
        help="Local cache directory for PaddleOCR models.",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR language.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.72,
        help="Minimum OCR confidence kept in the final result.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=1000,
        help="OCR tile width inside each detected parking region.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=800,
        help="OCR tile height inside each detected parking region.",
    )
    parser.add_argument(
        "--tile-step-x",
        type=int,
        default=900,
        help="Horizontal tile step.",
    )
    parser.add_argument(
        "--tile-step-y",
        type=int,
        default=700,
        help="Vertical tile step.",
    )
    parser.add_argument(
        "--tile-scale",
        type=float,
        default=2.2,
        help="Scale factor applied to each OCR tile before recognition.",
    )
    return parser.parse_args()


def configure_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"


def build_parking_mask(cleaned: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    return np.where(gray < 200, 255, 0).astype(np.uint8)


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        next_boxes: list[list[int]] = []
        while merged:
            current = merged.pop(0)
            cx1, cy1, cx2, cy2 = current
            keep_merging = True
            while keep_merging:
                keep_merging = False
                for index, other in enumerate(merged):
                    ox1, oy1, ox2, oy2 = other
                    horizontal_gap = max(0, max(cx1, ox1) - min(cx2, ox2))
                    vertical_gap = max(0, max(cy1, oy1) - min(cy2, oy2))
                    overlap_w = max(0, min(cx2, ox2) - max(cx1, ox1))
                    overlap_h = max(0, min(cy2, oy2) - max(cy1, oy1))
                    if (
                        overlap_w > 0
                        and overlap_h > 0
                        or horizontal_gap <= 90 and vertical_gap <= 60
                    ):
                        cx1 = min(cx1, ox1)
                        cy1 = min(cy1, oy1)
                        cx2 = max(cx2, ox2)
                        cy2 = max(cy2, oy2)
                        merged.pop(index)
                        keep_merging = True
                        changed = True
                        break
            next_boxes.append([cx1, cy1, cx2, cy2])
        merged = next_boxes
    return [tuple(box) for box in merged]


def detect_parking_regions(cleaned: np.ndarray) -> list[RegionBox]:
    mask = build_parking_mask(cleaned)
    height, width = mask.shape

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15)),
        iterations=2,
    )
    dilated = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 50000:
            continue
        if w < 220 or h < 80:
            continue
        if h < 120 and y > int(height * 0.85):
            continue
        if h > int(height * 0.42):
            continue
        if y > int(height * 0.95):
            continue
        if x < int(width * 0.12) and w < int(width * 0.20):
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.03:
            continue

        pad_x = max(12, int(w * 0.01))
        pad_y = max(12, int(h * 0.03))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(width, x + w + pad_x)
        bottom = min(height, y + h + pad_y)
        boxes.append((left, top, right, bottom))

    merged_boxes = merge_boxes(sorted(boxes, key=lambda item: (item[1], item[0])))
    regions = []
    for index, (left, top, right, bottom) in enumerate(
        sorted(merged_boxes, key=lambda item: (item[1], item[0])),
        start=1,
    ):
        regions.append(
            RegionBox(
                region_id=f"region_{index:02d}",
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        )
    return regions


def draw_region_overlay(image: np.ndarray, regions: list[RegionBox]) -> np.ndarray:
    overlay = image.copy()
    for region in regions:
        cv2.rectangle(
            overlay,
            (region.left, region.top),
            (region.right, region.bottom),
            (0, 180, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            region.region_id,
            (region.left + 8, max(region.top + 34, 36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def build_region_masked_image(image: np.ndarray, regions: list[RegionBox]) -> np.ndarray:
    masked = np.full_like(image, 255)
    for region in regions:
        masked[region.top : region.bottom, region.left : region.right] = image[
            region.top : region.bottom,
            region.left : region.right,
        ]
    return masked


def save_region_crops(image: np.ndarray, output_dir: Path, regions: list[RegionBox]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in regions:
        crop = image[region.top : region.bottom, region.left : region.right]
        save_image(output_dir / f"{region.region_id}.png", crop)


def build_positions(start: int, end: int, tile_size: int, step: int) -> list[int]:
    if end - start <= tile_size:
        return [start]
    positions = list(range(start, end - tile_size + 1, step))
    final_position = end - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def generate_tiles(
    region: RegionBox,
    tile_width: int,
    tile_height: int,
    step_x: int,
    step_y: int,
) -> list[dict]:
    x_positions = build_positions(region.left, region.right, tile_width, step_x)
    y_positions = build_positions(region.top, region.bottom, tile_height, step_y)
    tiles = []
    for top in y_positions:
        for left in x_positions:
            tiles.append(
                {
                    "region_id": region.region_id,
                    "left": left,
                    "top": top,
                    "right": min(region.right, left + tile_width),
                    "bottom": min(region.bottom, top + tile_height),
                }
            )
    return tiles


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
    quad = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if quad.shape[0] < 4:
        x_min, y_min = quad.min(axis=0)
        x_max, y_max = quad.max(axis=0)
        quad = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
    return quad


def quad_to_aabb(quad: np.ndarray) -> tuple[float, float, float, float]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def clean_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


@lru_cache(maxsize=2048)
def split_number_sequence(digits: str) -> tuple[str, ...]:
    if not digits:
        return ()
    if len(digits) <= 4:
        return (digits,)

    for width in (4, 3):
        if len(digits) < width * 2 or len(digits) % width != 0:
            continue
        chunks = tuple(digits[index : index + width] for index in range(0, len(digits), width))
        values = [int(chunk) for chunk in chunks]
        if width == 4 and not all(1000 <= value <= 1999 for value in values):
            continue
        if width == 3 and not all(100 <= value <= 999 for value in values):
            continue
        deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
        if deltas and max(abs(delta) for delta in deltas) <= 5 and sum(delta < 0 for delta in deltas) <= 1:
            return chunks

    @lru_cache(maxsize=None)
    def solve(index: int, prev_value: int) -> tuple[float, tuple[str, ...]]:
        if index >= len(digits):
            return 0.0, ()

        candidates: list[tuple[float, tuple[str, ...]]] = []

        if digits[index] == "1":
            score, tail = solve(index + 1, prev_value)
            candidates.append((score + 0.32, tail))

        for take, extra_penalty, strip_leading in (
            (4, 0.0, False),
            (3, 0.12, False),
            (4, 0.36, True),
            (2, 0.54, False),
        ):
            if index + take > len(digits):
                continue

            chunk = digits[index : index + take]
            if strip_leading:
                if not chunk.startswith("1"):
                    continue
                token = chunk[1:]
            else:
                token = chunk

            if not token:
                continue

            value = int(token)
            if value <= 0 or value > 1999:
                continue
            if len(token) == 4 and value < 1000:
                continue

            transition_penalty = extra_penalty
            if prev_value >= 0:
                delta = value - prev_value
                if delta == 1:
                    transition_penalty -= 0.05
                elif 0 <= delta <= 2:
                    transition_penalty += 0.04
                elif 3 <= delta <= 5:
                    transition_penalty += 0.24
                elif -1 <= delta < 0:
                    transition_penalty += 0.38
                else:
                    transition_penalty += 1.25
            else:
                if len(token) == 3:
                    transition_penalty -= 0.04
                elif len(token) != 4:
                    transition_penalty += 0.14

            score, tail = solve(index + take, value)
            candidates.append((score + transition_penalty, (token,) + tail))

        if not candidates:
            return 999.0, ()
        return min(
            candidates,
            key=lambda item: (
                item[0],
                -len(item[1]),
                sum(abs(len(token) - 3) for token in item[1]),
            ),
        )

    _, best = solve(0, -1)
    if best:
        return best

    chunks = []
    start = 0
    while start < len(digits):
        chunks.append(digits[start : start + 3])
        start += 3
    return tuple(chunk for chunk in chunks if chunk)


def expand_detected_text(text: str) -> list[str]:
    digits = clean_digits(text)
    if not digits:
        return []
    parts = [part.lstrip("0") or "0" for part in split_number_sequence(digits)]
    return [part for part in parts if 100 <= int(part) <= 1999 and len(part) >= 3]


def split_quad_evenly(quad: np.ndarray, count: int, index: int) -> np.ndarray:
    x1, y1, x2, y2 = quad_to_aabb(quad)
    width = max(x2 - x1, 1.0)
    left = x1 + width * index / count
    right = x1 + width * (index + 1) / count
    return np.array(
        [[left, y1], [right, y1], [right, y2], [left, y2]],
        dtype=np.float32,
    )


def should_keep_detection(
    bbox: tuple[float, float, float, float],
    region: RegionBox,
    score: float,
    min_score: float,
) -> bool:
    if score < min_score:
        return False
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width < 7 or height < 7:
        return False
    if width > region.width * 0.35 or height > region.height * 0.25:
        return False
    if x1 < region.left - 12 or x2 > region.right + 12:
        return False
    if y1 < region.top - 12 or y2 > region.bottom + 12:
        return False
    return True


def run_region_ocr(
    ocr,
    cleaned: np.ndarray,
    region: RegionBox,
    min_score: float,
    tile_width: int,
    tile_height: int,
    step_x: int,
    step_y: int,
    tile_scale: float,
) -> list[dict]:
    rows: list[dict] = []
    for tile in generate_tiles(region, tile_width, tile_height, step_x, step_y):
        crop = cleaned[tile["top"] : tile["bottom"], tile["left"] : tile["right"]]
        scaled = cv2.resize(
            crop,
            None,
            fx=tile_scale,
            fy=tile_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        results = ocr.predict(scaled, text_det_limit_side_len=max(scaled.shape[:2]))
        for box, text, score in iter_ocr_results(results):
            quad = normalize_box(box) / tile_scale
            quad[:, 0] += tile["left"]
            quad[:, 1] += tile["top"]
            bbox = quad_to_aabb(quad)
            if not should_keep_detection(bbox, region, score, min_score):
                continue

            slot_numbers = expand_detected_text(text)
            if not slot_numbers:
                continue

            for index, slot_number in enumerate(slot_numbers):
                split_quad = split_quad_evenly(quad, len(slot_numbers), index)
                rows.append(
                    {
                        "region_id": region.region_id,
                        "slot_number": slot_number,
                        "raw_text": text,
                        "score": score,
                        "quad": split_quad,
                        "bbox": quad_to_aabb(split_quad),
                    }
                )
    return rows


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


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: (-item["score"], int(item["slot_number"]))):
        cx = (row["bbox"][0] + row["bbox"][2]) / 2.0
        cy = (row["bbox"][1] + row["bbox"][3]) / 2.0
        duplicate = False
        for existing in kept:
            ex = (existing["bbox"][0] + existing["bbox"][2]) / 2.0
            ey = (existing["bbox"][1] + existing["bbox"][3]) / 2.0
            if row["slot_number"] != existing["slot_number"]:
                continue
            if iou(row, existing) > 0.20:
                duplicate = True
                break
            if abs(cx - ex) <= 18 and abs(cy - ey) <= 16:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    best_by_slot: dict[str, dict] = {}
    for row in kept:
        current = best_by_slot.get(row["slot_number"])
        if current is None or row["score"] > current["score"]:
            best_by_slot[row["slot_number"]] = row
    return sorted(
        best_by_slot.values(),
        key=lambda item: (
            int(item["slot_number"]),
            item["bbox"][1],
            item["bbox"][0],
        ),
    )


def draw_recognized_slots(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for row in rows:
        quad = row["quad"].astype(np.int32)
        cv2.polylines(annotated, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)
        x1, y1, _, _ = row["bbox"]
        label = row["slot_number"]
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        )
        tx = max(int(x1), 0)
        ty = max(int(y1) - 4, text_h + 6)
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
            0.50,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_slot_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region_id",
        "slot_number",
        "raw_text",
        "score",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "x3",
        "y3",
        "x4",
        "y4",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            quad = row["quad"].astype(int)
            center_x = int(round(np.mean(quad[:, 0])))
            center_y = int(round(np.mean(quad[:, 1])))
            writer.writerow(
                {
                    "region_id": row["region_id"],
                    "slot_number": row["slot_number"],
                    "raw_text": row["raw_text"],
                    "score": f"{row['score']:.4f}",
                    "center_x": center_x,
                    "center_y": center_y,
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


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_image(image_path)
    cleaned = keep_slot_lines_and_numbers(original)
    regions = detect_parking_regions(cleaned)

    stem = image_path.stem
    cleaned_path = output_dir / f"{stem}_cleaned.png"
    region_overlay_path = output_dir / f"{stem}_parking_regions.png"
    region_masked_path = output_dir / f"{stem}_parking_regions_masked.png"
    recognized_image_path = output_dir / f"{stem}_recognized_slots.png"
    csv_path = output_dir / f"{stem}_recognized_slots.csv"
    region_crops_dir = output_dir / f"{stem}_region_crops"

    save_image(cleaned_path, cleaned)
    save_image(region_overlay_path, draw_region_overlay(original, regions))
    save_image(region_masked_path, build_region_masked_image(original, regions))
    save_region_crops(original, region_crops_dir, regions)

    configure_env(Path(args.cache_dir))
    from paddleocr import PaddleOCR

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

    rows: list[dict] = []
    for region in regions:
        rows.extend(
            run_region_ocr(
                ocr=ocr,
                cleaned=cleaned,
                region=region,
                min_score=args.min_score,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                step_x=args.tile_step_x,
                step_y=args.tile_step_y,
                tile_scale=args.tile_scale,
            )
        )

    rows = deduplicate_rows(rows)
    recognized = draw_recognized_slots(original, rows)

    save_image(recognized_image_path, recognized)
    save_slot_csv(csv_path, rows)

    print(f"regions={len(regions)}")
    print(f"recognized_slots={len(rows)}")
    print(cleaned_path)
    print(region_overlay_path)
    print(region_masked_path)
    print(recognized_image_path)
    print(csv_path)


if __name__ == "__main__":
    main()
