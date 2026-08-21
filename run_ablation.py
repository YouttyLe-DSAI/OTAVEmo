import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import csv
import os

from dataset import AudioVisualDataset, collate_fn
from models import OTFusionModel, GRACEFusionModel, ConcatFusionModel, CrossAttentionFusionModel
from train import eval_epoch
from utils import load_model, checkpoint_exists

# Các mức độ desync mặc định muốn test (shift N frame)
# 0 = đồng bộ, dương = visual trễ, âm = visual sớm
DEFAULT_DESYNC_LEVELS = [-30, -20, -10, 0, 10, 20, 30]


def build_models(dims, beta):
    """4 model so sánh, key = tên hiển thị trong CSV, value = (tên checkpoint, model)."""
    return {
        'OT_Fusion (Ours)': ('ot_fusion', OTFusionModel(**dims, beta=beta)),
        'GRACE (beta=0)': ('grace', GRACEFusionModel(**dims)),
        'Concat': ('concat', ConcatFusionModel(**dims)),
        'Cross-Attention': ('cross_attn', CrossAttentionFusionModel(**dims)),
    }


def run_ablation(data_root, fold, checkpoint_dir='checkpoints', batch_size=32, hidden_dim=128,
                 beta=1.0, desync_levels=None, num_workers=2, out_csv=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu chạy Ablation Study trên: {device}")

    desync_levels = desync_levels or DEFAULT_DESYNC_LEVELS
    out_csv = out_csv or f'results/robustness_curve_fold{fold}.csv'

    # 1. Đánh giá trên tập TEST của fold này (fold == k theo FEATURE_CONTRACT.md)
    test_dataset = AudioVisualDataset(data_root, fold=fold, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    # 2. Khởi tạo model (dim/num_classes lấy từ meta.json) rồi load checkpoint tốt
    # nhất đã train bằng `train.py --model <key> --fold <k>`, tên file <model>_fold<k>.pt.
    # Nếu thiếu checkpoint nào thì cảnh báo và dùng trọng số khởi tạo ngẫu nhiên —
    # cột đó trong CSV sẽ không có ý nghĩa so sánh.
    dims = dict(audio_dim=test_dataset.audio_dim, visual_dim=test_dataset.visual_dim,
                hidden_dim=hidden_dim, num_classes=test_dataset.num_classes)

    models = {}
    for display_name, (ckpt_key, model) in build_models(dims, beta).items():
        ckpt_name = f"{ckpt_key}_fold{fold}"
        model = model.to(device)
        if checkpoint_exists(ckpt_name, save_dir=checkpoint_dir):
            model = load_model(model, ckpt_name, save_dir=checkpoint_dir, device=device)
            print(f"[{display_name}] Đã load checkpoint: {checkpoint_dir}/{ckpt_name}.pt")
        else:
            print(f"[{display_name}] CẢNH BÁO: chưa có checkpoint '{ckpt_name}.pt' — "
                  f"chạy 'python train.py --model {ckpt_key} --fold {fold} --data_root ...' trước. "
                  f"Dùng tạm trọng số random.")
        model.eval()
        models[display_name] = model

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    with open(out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi header
        header = ['Desync_Frames'] + list(models.keys())
        writer.writerow(header)

        # 3. Chạy đánh giá
        for shift in desync_levels:
            print(f"\n--- Đánh giá với mức độ Desync = {shift} frames ---")
            row = [shift]

            for model_name, model in models.items():
                loss, acc = eval_epoch(model, test_loader, criterion, device, desync_frames=shift)
                print(f"{model_name:<20}: Acc = {acc:.2f}%")
                row.append(f"{acc:.2f}")

            writer.writerow(row)

    print(f"\n[Hoàn thành] Đã xuất kết quả ra {out_csv}. Bạn có thể dùng file này để vẽ Robustness Curve.")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Robustness curve: accuracy vs. mức độ desync, cho cả 4 model")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Thư mục feature theo docs/FEATURE_CONTRACT.md")
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Fold dùng làm tập test — phải khớp với fold đã train checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help="Phải khớp --hidden_dim lúc train, nếu không state_dict sẽ không load được")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Beta của OTFusionModel — phải khớp lúc train")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--desync_levels', type=int, nargs='+', default=None,
                        help=f"Các mức shift frame cần test (mặc định: {DEFAULT_DESYNC_LEVELS})")
    parser.add_argument('--out', type=str, default=None,
                        help="Đường dẫn CSV output (mặc định results/robustness_curve_fold<k>.csv)")
    args = parser.parse_args()

    run_ablation(args.data_root, fold=args.fold, checkpoint_dir=args.checkpoint_dir,
                 batch_size=args.batch_size, hidden_dim=args.hidden_dim, beta=args.beta,
                 desync_levels=args.desync_levels, num_workers=args.num_workers, out_csv=args.out)


if __name__ == '__main__':
    main()
