"""
Tạo một dataset GIẢ nhưng đúng 100% docs/FEATURE_CONTRACT.md.

Mục đích DUY NHẤT: smoke test pipeline train → checkpoint → ablation → CSV khi
feature thật của DFEW/MAFW chưa xong. Feature ở đây là `torch.randn`, KHÔNG mang
bất kỳ thông tin nào — mọi con số accuracy chạy trên fixture này là vô nghĩa về
mặt khoa học, tuyệt đối không đưa vào paper.

    python make_fixture_dataset.py --out /tmp/fixture_mafw
    python docs/check_dataset_format.py /tmp/fixture_mafw --full
"""
import argparse
import csv
import json
import os
import shutil

import torch

# 11 lớp MAFW theo đúng thứ tự trong docs/FEATURE_CONTRACT.md mục 3.
# (Contract có ghi chú: thứ tự này vẫn cần đối chiếu với file split gốc trước khi
# chốt cho dataset thật — ở đây chỉ dùng để fixture giống hình dạng thật.)
MAFW_CLASS_NAMES = [
    "anger", "disgust", "fear", "happiness", "neutral", "sadness",
    "surprise", "contempt", "anxiety", "helplessness", "disappointment",
]

# Trọng số phân bố lớp lệch nhau, mô phỏng độ mất cân bằng thật của DFEW/MAFW
# (lớp phổ biến nhiều mẫu gấp hàng chục lần lớp hiếm) để còn test --class_weighted.
CLASS_SKEW = [8, 1, 3, 12, 10, 9, 4, 1, 3, 2, 2]


def make_fixture(out_dir, num_clips=60, audio_dim=768, visual_dim=512,
                 min_frames=16, max_frames=64, seed=0, overwrite=False):
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} đã tồn tại. Dùng --overwrite nếu muốn xoá và tạo lại.")
        shutil.rmtree(out_dir)

    audio_dir = os.path.join(out_dir, 'audio')
    visual_dir = os.path.join(out_dir, 'visual')
    os.makedirs(audio_dir)
    os.makedirs(visual_dir)

    g = torch.Generator().manual_seed(seed)
    num_classes = len(MAFW_CLASS_NAMES)

    # Gán nhãn: đảm bảo mỗi lớp có ít nhất 1 mẫu (tránh lớp trống làm class weight
    # thành 0), phần còn lại bốc theo phân bố lệch ở trên.
    labels = list(range(num_classes))
    remaining = num_clips - num_classes
    if remaining < 0:
        raise ValueError(f"--num_clips phải >= {num_classes} để mỗi lớp có ít nhất 1 mẫu")
    weights = torch.tensor(CLASS_SKEW, dtype=torch.float)
    labels += torch.multinomial(weights, remaining, replacement=True, generator=g).tolist()

    # Xáo thứ tự rồi chia fold kiểu round-robin: 5 fold đều có mẫu, và nhãn rải
    # tương đối đều qua các fold (giống split chính thức đã stratify sẵn).
    perm = torch.randperm(num_clips, generator=g).tolist()
    labels = [labels[i] for i in perm]

    rows = []
    for i, label in enumerate(labels):
        clip_id = f"fixture_{i:04d}"
        fold = (i % 5) + 1

        # T tự do theo từng clip, và T_a khác T_v — đúng tinh thần contract
        # (không pad sẵn), đồng thời ép pipeline phải xử lý đúng chuyện lệch độ dài.
        t_a = int(torch.randint(min_frames, max_frames + 1, (1,), generator=g))
        t_v = int(torch.randint(min_frames, max_frames + 1, (1,), generator=g))

        torch.save(torch.randn(t_a, audio_dim, generator=g, dtype=torch.float32),
                   os.path.join(audio_dir, f'{clip_id}.pt'))
        torch.save(torch.randn(t_v, visual_dim, generator=g, dtype=torch.float32),
                   os.path.join(visual_dir, f'{clip_id}.pt'))

        rows.append({'clip_id': clip_id, 'label': label, 'fold': fold})

    with open(os.path.join(out_dir, 'labels.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['clip_id', 'label', 'fold'])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "dataset_name": "fixture_mafw_FAKE",
        "num_classes": num_classes,
        "class_names": MAFW_CLASS_NAMES,
        "audio_dim": audio_dim,
        "audio_extractor": "FAKE torch.randn (khong phai wav2vec2)",
        "visual_dim": visual_dim,
        "visual_extractor": "FAKE torch.randn (khong phai CLIP/VideoMAE)",
        "num_clips": num_clips,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    per_class = [0] * num_classes
    per_fold = [0] * 5
    for r in rows:
        per_class[r['label']] += 1
        per_fold[r['fold'] - 1] += 1

    print(f"Đã tạo fixture GIẢ tại: {os.path.abspath(out_dir)}")
    print(f"  {num_clips} clip, audio_dim={audio_dim}, visual_dim={visual_dim}, "
          f"T trong khoảng [{min_frames}, {max_frames}]")
    print("  phân bố lớp: " + ', '.join(f"{n}={c}" for n, c in zip(MAFW_CLASS_NAMES, per_class)))
    print("  phân bố fold: " + ', '.join(f"fold{i + 1}={c}" for i, c in enumerate(per_fold)))
    print("\nLƯU Ý: đây là dữ liệu NGẪU NHIÊN, chỉ để kiểm tra pipeline chạy được.")
    print(f"Kiểm tra format: python docs/check_dataset_format.py {out_dir} --full")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Sinh dataset giả đúng format FEATURE_CONTRACT.md")
    parser.add_argument('--out', type=str, required=True, help="Thư mục output")
    parser.add_argument('--num_clips', type=int, default=60)
    parser.add_argument('--audio_dim', type=int, default=768, help="Khớp wav2vec2-base")
    parser.add_argument('--visual_dim', type=int, default=512, help="Khớp CLIP ViT-B/32")
    parser.add_argument('--min_frames', type=int, default=16)
    parser.add_argument('--max_frames', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    make_fixture(args.out, num_clips=args.num_clips, audio_dim=args.audio_dim,
                 visual_dim=args.visual_dim, min_frames=args.min_frames,
                 max_frames=args.max_frames, seed=args.seed, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
