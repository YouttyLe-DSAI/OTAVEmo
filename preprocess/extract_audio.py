"""
Bước 2 của preprocessing: trích audio từ mp4 gốc ra WAV 16 kHz mono.

16 kHz mono là định dạng đầu vào chuẩn của emotion2vec / wav2vec2.
DFEW có clip 44.1 kHz lẫn 48 kHz, đều stereo -> ffmpeg lo cả resample lẫn downmix.
(MAFW còn có clip 5.1 và mono, cùng lệnh này vẫn chuẩn hoá được.)

Mặc định chỉ xử lý clip nằm trong split 5-fold (in_split=1) để đỡ thời gian và đĩa.

Cách chạy:
    python preprocess/extract_audio.py
    python preprocess/extract_audio.py --all        # cả 16,372 clip
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

TARGET_SR = 16000


def extract_one(job):
    """Trích 1 clip. Trả (clip_id, ok, thông điệp lỗi)."""
    clip_id, src, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 44:   # đã có, bỏ qua
        return clip_id, True, "skip"
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-i", src,
        "-vn",                       # bỏ luồng hình
        "-ac", "1",                  # downmix về mono
        "-ar", str(TARGET_SR),       # resample 16 kHz
        "-sample_fmt", "s16",        # PCM 16-bit
        "-f", "wav", dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return clip_id, False, "timeout"
    if proc.returncode != 0:
        return clip_id, False, proc.stderr.strip()[:150]
    if not os.path.exists(dst) or os.path.getsize(dst) <= 44:
        return clip_id, False, "wav rỗng"
    return clip_id, True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfew-root", default="../Dataset/DFEW")
    ap.add_argument("--manifest", default="../Preprocessing_PAPER1/manifest/dfew_manifest.csv")
    ap.add_argument("--out-dir", default="../Preprocessing_PAPER1/audio_16k")
    ap.add_argument("--all", action="store_true",
                    help="Trích cả clip ngoài split 5-fold")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = os.path.abspath(args.dfew_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.manifest):
        sys.exit(f"Chưa có manifest: {args.manifest}\n"
                 f"Chạy `python preprocess/build_manifest.py` trước.")

    jobs, skipped_no_audio = [], []
    with open(args.manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            if not args.all and row["in_split"] != "1":
                continue
            if row["has_audio"] != "1":
                skipped_no_audio.append(row["clip_id"])
                continue
            src = os.path.join(root, row["mp4_path"])
            dst = os.path.join(out_dir, f"{int(row['clip_id']):05d}.wav")
            jobs.append((row["clip_id"], src, dst))

    print(f"Sẽ trích {len(jobs):,} clip -> {out_dir}  ({args.workers} tiến trình)")
    if skipped_no_audio:
        print(f"  bỏ qua {len(skipped_no_audio):,} clip không có audio: "
              f"{skipped_no_audio[:10]}")

    ok = skipped = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (cid, good, msg) in enumerate(
                pool.map(extract_one, jobs, chunksize=16), 1):
            if good:
                ok += 1
                skipped += (msg == "skip")
            else:
                failures.append((cid, msg))
            if done % 1000 == 0:
                print(f"  {done:,}/{len(jobs):,}  (lỗi: {len(failures)})")

    print(f"\nXong: {ok:,}/{len(jobs):,} thành công "
          f"(trong đó {skipped:,} đã có sẵn nên bỏ qua), {len(failures):,} lỗi")
    if failures:
        log = os.path.join(os.path.dirname(out_dir), "logs", "extract_audio_failures.txt")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            for cid, msg in failures:
                fh.write(f"{cid}\t{msg}\n")
        print(f"  chi tiết lỗi: {log}")
        for cid, msg in failures[:5]:
            print(f"    {cid}: {msg}")


if __name__ == "__main__":
    main()
