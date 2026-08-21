# Feature Contract — OT-AV Emotion (DFEW / MAFW)

Mục đích: 3 người (bạn + 2 người xử lý DFEW/MAFW) xuất feature độc lập, miễn
đúng format này thì `dataset.py`/`train.py` load thẳng được, không cần biết
ai dùng model/pipeline gì để trích.

## Cấu trúc thư mục bắt buộc

```
<dataset_root>/            # vd: data/mafw/ hoặc data/dfew/
├── audio/
│   ├── <clip_id>.pt
│   └── ...
├── visual/
│   ├── <clip_id>.pt
│   └── ...
├── labels.csv
└── meta.json
```

## 1. File feature — `audio/<clip_id>.pt`, `visual/<clip_id>.pt`

- Lưu bằng `torch.save(tensor, path)`.
- Tensor **2 chiều** `(T, D)`:
  - `T` = số frame, **tự do theo từng clip** — KHÔNG pad trước, KHÔNG cắt cho bằng nhau.
  - `D` = feature dim, **cố định cho toàn bộ dataset** (khai báo trong `meta.json`).
- `dtype = torch.float32`.
- Không NaN / Inf.
- Không cần lưu mask riêng — mask được suy ra từ `T` thật lúc collate (pad + mask tự động).
- `clip_id`: chuỗi, phải khớp **chính xác** (không đuôi file, không khoảng trắng) với cột `clip_id` trong `labels.csv` VÀ với ID gốc trong file nhãn chính thức của DFEW/MAFW.

## 2. `labels.csv`

Cột bắt buộc: `clip_id,label,fold`

| cột | ý nghĩa |
|---|---|
| `clip_id` | khớp tên file `.pt` (không đuôi mở rộng) |
| `label` | số nguyên bắt đầu từ 0, theo **đúng thứ tự** khai báo trong `meta.json["class_names"]` |
| `fold` | số nguyên 1–5, chỉ số fold mà clip này thuộc **tập TEST** theo protocol 5-fold chính thức |

Cách map fold: với DFEW, clip nằm trong `single_testset_k.csv` gốc → `fold=k`.
Với MAFW làm tương tự theo file split gốc đi kèm dataset. Khi train fold `k`:
train = tất cả clip có `fold != k`, test = `fold == k`.

## 3. `meta.json`

```json
{
  "dataset_name": "mafw",
  "num_classes": 11,
  "class_names": ["...", "..."],
  "audio_dim": 768,
  "audio_extractor": "facebook/wav2vec2-base",
  "visual_dim": 512,
  "visual_extractor": "CLIP ViT-B/32, 40 frame/clip",
  "num_clips": 9172
}
```

**QUAN TRỌNG**: `class_names` phải theo **đúng thứ tự index** dùng trong cột
`label` — copy nguyên văn thứ tự trong file nhãn gốc của dataset, không tự
sắp lại theo alphabet. Nhầm thứ tự = train sai nhãn mà code không hề báo lỗi.

Tham khảo (đối chiếu lại với file nhãn gốc khi tải dataset để chắc chắn):

- **DFEW** (7 lớp, theo mapping chính thức của tác giả, 0-indexed):
  `["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]`
- **MAFW** (11 lớp, theo thứ tự liệt kê trong paper gốc — cần đối chiếu numeric
  ID trong file split chính thức trước khi chốt, vì paper có thể chỉ liệt kê
  mô tả chứ không đảm bảo đúng thứ tự index trong CSV):
  `["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
  "contempt", "anxiety", "helplessness", "disappointment"]`

## 4. Checklist trước khi báo "xong" cho phần của mình

- [ ] Mọi `clip_id` trong `labels.csv` đều có đúng 1 file trong `audio/` và 1 file trong `visual/`
- [ ] Mọi tensor đều `(T, D)`, `D` giống nhau trong toàn bộ `audio/` (và tương tự cho `visual/`)
- [ ] `class_names` đã đối chiếu với file nhãn gốc, không phải suy đoán
- [ ] Chạy `python check_dataset_format.py <dataset_root>` và thấy `RESULT: PASS`
