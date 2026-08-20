import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import os

from dataset import AudioVisualDataset, collate_fn
from models import build_model, count_params
from train import eval_epoch
from utils import load_model, checkpoint_exists

def run_ablation(checkpoint_dir='checkpoints', beta=1.0, hidden_dim=128,
                 mult_hidden_dim=64, mult_layers=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu chạy Ablation Study trên: {device}")

    # 1. Chuẩn bị dataset (Chỉ cần validation/test set cho phần ablation)
    val_dataset = AudioVisualDataset(split='test', num_samples=500)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()

    # 2. Khởi tạo model qua build_model (CÙNG cấu hình với train.py) rồi load
    # checkpoint tốt nhất. Thiếu checkpoint thì DỪNG HẲN — chạy tiếp với trọng số
    # ngẫu nhiên sẽ tạo ra bảng số trông như thật nhưng vô nghĩa.
    display_of = {
        'ot_fusion': 'OT_Fusion (Ours)',
        'grace': 'GRACE (beta=0)',
        'concat': 'Concat',
        'cross_attn': 'Cross-Attention',
    }

    models, n_params = {}, {}
    missing = [k for k in display_of if not checkpoint_exists(k, save_dir=checkpoint_dir)]
    if missing:
        raise FileNotFoundError(
            "Thiếu checkpoint cho: " + ", ".join(missing) + ".\n"
            "Chạy trước: " + "  ".join(f"python train.py --model {k}" for k in missing))

    for ckpt_name, display_name in display_of.items():
        model = build_model(ckpt_name, hidden_dim=hidden_dim, beta=beta,
                            mult_hidden_dim=mult_hidden_dim, mult_layers=mult_layers).to(device)
        n_params[display_name] = count_params(model)
        model = load_model(model, ckpt_name, save_dir=checkpoint_dir, device=device)
        print(f"[{display_name}] checkpoint OK — {n_params[display_name]:,} tham số")
        model.eval()
        models[display_name] = model

    # Cảnh báo nếu số tham số lệch nhau quá nhiều -> so sánh không công bằng
    lo, hi = min(n_params.values()), max(n_params.values())
    if hi / lo > 1.5:
        print(f"\n⚠ CẢNH BÁO: số tham số lệch {hi/lo:.1f}x giữa các model "
              f"({lo:,} .. {hi:,}). Kết quả so sánh sẽ bị phản biện.\n")

    # Các mức độ desync muốn test (Shift N frames)
    # 0 = đồng bộ, dương = visual trễ, âm = visual sớm
    desync_levels = [-30, -20, -10, 0, 10, 20, 30] 
    
    # Mở file CSV để ghi kết quả
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/robustness_curve.csv'
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Ghi header + số tham số của từng model (để bảng kết quả tự chứng minh
        # so sánh công bằng, không phải tin lời)
        header = ['Desync_Frames'] + list(models.keys())
        writer.writerow(header)
        writer.writerow(['n_params'] + [n_params[k] for k in models])
        
        # 3. Chạy đánh giá
        for shift in desync_levels:
            print(f"\n--- Đánh giá với mức độ Desync = {shift} frames ---")
            row = [shift]
            
            for model_name, model in models.items():
                # Thực hiện evaluate
                loss, acc = eval_epoch(model, val_loader, criterion, device, desync_frames=shift)
                print(f"{model_name:<20}: Acc = {acc:.2f}%")
                row.append(f"{acc:.2f}")
                
            writer.writerow(row)
            
    print(f"\n[Hoàn thành] Đã xuất kết quả ra {csv_file}. Bạn có thể dùng file này để vẽ Robustness Curve.")

if __name__ == '__main__':
    run_ablation()
