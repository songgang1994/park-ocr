import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise parking plan images and keep slot wireframes and numbers."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input image paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="park_ocr/output",
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix",
        default="cleaned",
        help="Filename prefix for generated images.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def remove_colored_annotations(hsv: np.ndarray) -> np.ndarray:
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    vivid = (saturation > 65) & (value > 110)
    red = vivid & ((hue <= 10) | (hue >= 170))
    magenta = vivid & (hue >= 135) & (hue <= 169)
    green = vivid & (hue >= 35) & (hue <= 95)
    cyan = vivid & (hue >= 85) & (hue <= 120)

    return red | magenta | green | cyan


def remove_building_fill(image: np.ndarray) -> np.ndarray:
    bgr = image.astype(np.int16)
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]

    tan_fill = (
        (red > 110)
        & (green > 90)
        & (blue > 70)
        & (red - green > 10)
        & (green - blue > 8)
        & (red < 235)
        & (green < 220)
        & (blue < 205)
    )

    dark_tan = (
        (red > 70)
        & (green > 45)
        & (blue > 25)
        & (red - green > 12)
        & (green - blue > 8)
        & (red < 170)
        & (green < 135)
        & (blue < 110)
    )

    return tan_fill | dark_tan


def filter_components(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros(binary.shape, dtype=np.uint8)
    height, width = binary.shape

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 3:
            continue
        if y > int(height * 0.93):
            continue
        if area > 30000:
            continue

        fill_ratio = area / float(w * h)

        if min(w, h) <= 2 and max(w, h) > 150:
            continue
        if area > 6000 and fill_ratio > 0.16:
            continue
        if area > 400 and fill_ratio > 0.82:
            continue
        if w > int(width * 0.55) and h < 10:
            continue
        if h > int(height * 0.22) and w < 10:
            continue

        component = labels[y : y + h, x : x + w] == label
        kept[y : y + h, x : x + w][component] = 255

    return kept


def keep_slot_lines_and_numbers(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    annotation_mask = remove_colored_annotations(hsv)
    building_mask = remove_building_fill(image)
    building_mask = (
        cv2.dilate(
            building_mask.astype(np.uint8) * 255,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        > 0
    )

    work = image.copy()
    work[annotation_mask | building_mask] = 255

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    combined = np.where(gray < 245, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = filter_components(cleaned)

    result = np.full_like(image, 255)
    result[cleaned > 0] = (0, 0, 0)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        image = load_image(input_path)
        cleaned = keep_slot_lines_and_numbers(image)
        output_path = output_dir / f"{input_path.stem}_{args.prefix}.png"
        save_image(output_path, cleaned)
        print(output_path)


if __name__ == "__main__":
    main()
