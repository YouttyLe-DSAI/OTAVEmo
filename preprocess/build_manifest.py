"""
Bước 1 của preprocessing: quét toàn bộ DFEW và dựng manifest dùng chung.

Xuất ra (mặc định vào ../Preprocessing_PAPER1/manifest/):
  dfew_manifest.csv       — 1 dòng / clip, các trường vô hướng
  dfew_frame_indices.npz  — clip_id -> mảng chỉ số frame GỐC (int32)

Chỉ số frame gốc là mấu chốt: ảnh `x_y.jpg` có `y` = số thứ tự frame trong `x.mp4`,
và frame không detect được mặt bị bỏ nên `y` KHÔNG liên tục. Nhờ giữ được `y`,
timestamp thật tra ngược được:   t(frame y) = (y - 1) / fps
Đây là điều kiện cần để xây cost matrix của OT theo GIÂY thay vì theo chỉ số mảng.

Cách chạy:
    python preprocess/build_manifest.py --dfew-root ../Dataset/DFEW
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np

EMOTIONS = {1: "happy", 2: "sad", 3: "neutral", 4: "angry",
            5: "surprise", 6: "disgust", 7: "fear"}


def probe(mp4_path):
    """Đọc metadata video/audio bằng ffprobe. Trả về dict, hoặc dict rỗng nếu lỗi."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", mp4_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        streams = json.loads(out).get("streams", [])
    except Exception as exc:                      # ffprobe lỗi / file hỏng
        return {"probe_error": str(exc)[:120]}

    vid = next((s for s in streams if s["codec_type"] == "video"), None)
    aud = next((s for s in streams if s["codec_type"] == "audio"), None)
    if vid is None:
        return {"probe_error": "no video stream"}

    num, den = vid["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0

    rec = {
        "fps": round(fps, 6),
        "width": vid["width"],
        "height": vid["height"],
        "n_frames_video": int(vid.get("nb_frames", 0)),
        "duration_video": round(float(vid.get("duration", 0.0)), 6),
        "has_audio": int(aud is not None),
        "probe_error": "",
    }
    if aud is not None:
        rec.update({
            "audio_codec": aud["codec_name"],
            "audio_sr": int(aud["sample_rate"]),
            "audio_channels": int(aud["channels"]),
            "duration_audio": round(float(aud.get("duration", 0.0)), 6),
        })
        rec["av_offset"] = round(rec["duration_audio"] - rec["duration_video"], 6)
    else:
        rec.update({"audio_codec": "", "audio_sr": 0, "audio_channels": 0,
                    "duration_audio": 0.0, "av_offset": 0.0})
    return rec


def _probe_one(item):
    clip_id, path = item
    return clip_id, probe(path)


def index_videos(root):
    """clip_id (int) -> đường dẫn mp4."""
    out = {}
    for p in glob(os.path.join(root, "Clip", "original", "part_*", "*.mp4")):
        out[int(os.path.splitext(os.path.basename(p))[0])] = p
    return out


def index_face_dirs(root):
    """clip_id (int) -> thư mục chứa frame mặt đã align."""
    out = {}
    pattern = os.path.join(root, "Clip", "clip_224x224", "clip_224x224_part_*", "*")
    for p in glob(pattern):
        if os.path.isdir(p):
            out[int(os.path.basename(p))] = p
    return out


def read_frame_indices(face_dir):
    """Đọc chỉ số frame GỐC từ tên file `<clip>_<idx>.jpg`, trả mảng đã sắp xếp."""
    idx = []
    for name in os.listdir(face_dir):
        if not name.lower().endswith(".jpg"):
            continue
        try:
            idx.append(int(os.path.splitext(name)[0].split("_")[1]))
        except (IndexError, ValueError):
            continue
    return np.array(sorted(idx), dtype=np.int32)


def load_splits(root):
    """clip_id -> (label, fold_test). fold_test = fold mà clip nằm ở TẬP TEST."""
    labels, fold_of = {}, {}
    for fold in range(1, 6):
        for subset in ("train", "test"):
            path = os.path.join(root, "EmoLabel_DataSplit",
                                f"{subset}(single-labeled)", f"set_{fold}.csv")
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    cid = int(row["video_name"])
                    labels[cid] = int(row["label"])
                    if subset == "test":
                        fold_of[cid] = fold
    return labels, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfew-root", default="../Dataset/DFEW")
    ap.add_argument("--out-dir", default="../Preprocessing_PAPER1/manifest")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = os.path.abspath(args.dfew_root)
    if not os.path.isdir(root):
        sys.exit(f"Không tìm thấy thư mục DFEW: {root}")
    os.makedirs(args.out_dir, exist_ok=True)

    videos = index_videos(root)
    face_dirs = index_face_dirs(root)
    labels, fold_of = load_splits(root)
    print(f"mp4: {len(videos):,} | thư mục frame mặt: {len(face_dirs):,} | "
          f"clip có nhãn: {len(labels):,}")

    # --- ffprobe song song trên toàn bộ clip ---
    items = sorted(videos.items())
    print(f"Đang probe {len(items):,} clip bằng {args.workers} tiến trình...")
    probes = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (cid, rec) in enumerate(pool.map(_probe_one, items, chunksize=32), 1):
            probes[cid] = rec
            if done % 2000 == 0:
                print(f"  {done:,}/{len(items):,}")

    # --- đọc chỉ số frame + ghép mọi thứ lại ---
    fields = ["clip_id", "in_split", "label", "emotion", "fold_test",
              "fps", "width", "height", "n_frames_video", "duration_video",
              "n_frames_face", "frame_idx_first", "frame_idx_last", "frame_coverage",
              "has_audio", "audio_codec", "audio_sr", "audio_channels",
              "duration_audio", "av_offset", "mp4_path", "face_dir", "probe_error"]

    rows, frame_idx_store, n_err = [], {}, 0
    for cid, _ in items:
        rec = probes[cid]
        if rec.get("probe_error"):
            n_err += 1
        face_dir = face_dirs.get(cid)
        fidx = read_frame_indices(face_dir) if face_dir else np.array([], dtype=np.int32)
        if fidx.size:
            frame_idx_store[str(cid)] = fidx

        nfv = rec.get("n_frames_video", 0)
        row = {
            "clip_id": cid,
            "in_split": int(cid in labels),
            "label": labels.get(cid, 0),
            "emotion": EMOTIONS.get(labels.get(cid, 0), ""),
            "fold_test": fold_of.get(cid, 0),
            "n_frames_face": int(fidx.size),
            "frame_idx_first": int(fidx[0]) if fidx.size else 0,
            "frame_idx_last": int(fidx[-1]) if fidx.size else 0,
            # tỉ lệ frame giữ lại được so với video gốc
            "frame_coverage": round(fidx.size / nfv, 4) if nfv else 0.0,
            "mp4_path": os.path.relpath(videos[cid], root),
            "face_dir": os.path.relpath(face_dir, root) if face_dir else "",
        }
        row.update({k: rec.get(k, "") for k in
                    ["fps", "width", "height", "n_frames_video", "duration_video",
                     "has_audio", "audio_codec", "audio_sr", "audio_channels",
                     "duration_audio", "av_offset", "probe_error"]})
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "dfew_manifest.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = os.path.join(args.out_dir, "dfew_frame_indices.npz")
    np.savez_compressed(npz_path, **frame_idx_store)

    in_split = [r for r in rows if r["in_split"]]
    no_audio = [r for r in rows if not r["has_audio"]]
    print(f"\nĐã ghi {csv_path}  ({len(rows):,} dòng)")
    print(f"Đã ghi {npz_path}  ({len(frame_idx_store):,} clip có frame)")
    print(f"  clip trong split 5-fold : {len(in_split):,}")
    print(f"  clip KHÔNG có audio     : {len(no_audio):,}")
    print(f"  clip probe lỗi          : {n_err:,}")
    if no_audio[:5]:
        print("  ví dụ clip thiếu audio :", [r["clip_id"] for r in no_audio[:10]])


if __name__ == "__main__":
    main()
