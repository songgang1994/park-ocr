import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


REGION_OFFSETS = {
    "phase1": (4100, 260),
    "phase2": (1650, 1750),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map split OCR CSV results back to the original image."
    )
    parser.add_argument("--image", required=True, help="Original image path.")
    parser.add_argument("--output-image", required=True, help="Annotated image path.")
    parser.add_argument("--output-csv", required=True, help="Merged CSV path.")
    parser.add_argument(
        "--phase1-csv",
        default="park_ocr/output/phase1_numbers_test.csv",
        help="Phase1 OCR CSV.",
    )
    parser.add_argument(
        "--phase2-csv",
        default="park_ocr/output/phase2_numbers_test.csv",
        help="Phase2 OCR CSV.",
    )
    return parser.parse_args()


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


def load_rows(path: Path, region: str) -> list[dict]:
    left, top = REGION_OFFSETS[region]
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            quad = np.array(
                [
                    [int(row["x1"]) + left, int(row["y1"]) + top],
                    [int(row["x2"]) + left, int(row["y2"]) + top],
                    [int(row["x3"]) + left, int(row["y3"]) + top],
                    [int(row["x4"]) + left, int(row["y4"]) + top],
                ],
                dtype=np.int32,
            )
            rows.append(
                {
                    "region": region,
                    "text": row["text"],
                    "score": float(row["score"]),
                    "quad": quad,
                }
            )
    return rows


def draw_annotations(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for row in rows:
        quad = row["quad"]
        cv2.polylines(annotated, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

        x = int(np.min(quad[:, 0]))
        y = int(np.min(quad[:, 1]))
        label = row["text"]
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tx = max(x, 0)
        ty = max(y - 6, text_h + 6)
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
            quad = row["quad"]
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


def main() -> None:
    args = parse_args()

    rows = []
    rows.extend(load_rows(Path(args.phase1_csv), "phase1"))
    rows.extend(load_rows(Path(args.phase2_csv), "phase2"))
    rows.sort(key=lambda item: (int(np.min(item["quad"][:, 1])), int(np.min(item["quad"][:, 0]))))

    image = load_image(Path(args.image))
    annotated = draw_annotations(image, rows)

    save_image(Path(args.output_image), annotated)
    save_csv(Path(args.output_csv), rows)

    print(f"kept={len(rows)}")
    print(args.output_image)
    print(args.output_csv)


if __name__ == "__main__":
    main()
