# Park OCR 项目文档

## 启动命令文档

```powershell
# 车位线条/编号清理
python park_ocr\keep_parking_slots.py --inputs park_ocr\20260306154854_3_441.jpg

# 一体化：区荐主流程）
python park_ocr\extract_parking_slots.py --image park_ocr\20260306154854_3_441.jpg

# 分区域 OCR（固定区域坐标），并输出标注图 + CSV
python park_ocr\annotate_split_numbers.py --image park_ocr\20260306154854_3_441.jpg --output-image park_ocr\output\phase_numbers.png --output-csv park_ocr\output\phase_numbers.csv

# 合并分区域 CSV 回原图坐标
python park_ocr\compose_split_annotations.py --image park_ocr\20260306154854_3_441.jpg --output-image park_ocr\output\merged.png --output-csv park_ocr\output\merged.csv
```

## 参数文档

### park_ocr\keep_parking_slots.py

- `--inputs`：必填。输入图片路径列表（支持多个）。
- `--output-dir`：默认 `park_ocr/output`。输出目录。
- `--prefix`：默认 `cleaned`。输出文件名前缀。

### park_ocr\extract_parking_slots.py

- `--image`：必填。输入车位图。
- `--output-dir`：默认 `park_ocr/output`。输出目录。
- `--cache-dir`：默认 `park_ocr/.paddlex_cache`。PaddleOCR 模型缓存目录。
- `--lang`：默认 `ch`。OCR 语言。
- `--min-score`：默认 `0.72`。OCR 最小置信度。
- `--tile-width`：默认 `1000`。区域内 OCR tile 宽。
- `--tile-height`：默认 `800`。区域内 OCR tile 高。
- `--tile-step-x`：默认 `900`。tile 横向步进。
- `--tile-step-y`：默认 `700`。tile 纵向步进。
- `--tile-scale`：默认 `2.2`。OCR 前缩放倍率。

### park_ocr\annotate_split_numbers.py

- `--image`：必填。输入图片。
- `--output-image`：必填。输出标注图片。
- `--output-csv`：必填。输出 OCR CSV。
- `--cache-dir`：默认 `park_ocr/.paddlex_cache`。PaddleOCR 模型缓存目录。
- `--lang`：默认 `ch`。OCR 语言。
- `--min-score`：默认 `0.45`。OCR 最小置信度。
- `--tile-size`：默认 `1600`。tile 尺寸。
- `--tile-stride`：默认 `1200`。tile 步长。
- `--tile-scale`：默认 `2.0`。OCR 前缩放倍率。

### park_ocr\compose_split_annotations.py

- `--image`：必填。原始图片。
- `--output-image`：必填。输出合并标注图。
- `--output-csv`：必填。输出合并 CSV。
- `--phase1-csv`：默认 `park_ocr/output/phase1_numbers_test.csv`。phase1 OCR CSV。
- `--phase2-csv`：默认 `park_ocr/output/phase2_numbers_test.csv`。phase2 OCR CSV。

## PaddleX GPU Notes

- Default OCR engine uses PaddleX pipeline `OCR` and its default config.
- Default device is `gpu`. Use `--device cpu` to force CPU.
- `--use-doc-preprocessor` and `--use-textline-orientation` only override defaults when provided.
- `--lang` is kept for compatibility but is not used by PaddleX OCR.
