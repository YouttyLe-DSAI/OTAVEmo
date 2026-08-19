import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import os

from dataset import AudioVisualDataset, collate_fn
from models import OTFusionModel, GRACEFusionModel, ConcatFusionModel, CrossAttentionFusionModel
from train import eval_epoch
from utils import load_model, checkpoint_exists

def run_ablation(checkpoint_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu chạy Ablation Study trên: {device}")

    # 1. Chuẩn bị dataset (Chỉ cần validation/test set cho phần ablation)
    val_dataset = AudioVisualDataset(split='test', num_samples=500)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()

    # 2. Khởi tạo model rồi load checkpoint tốt nhất đã train (bằng train.py --model <key>).
    # Nếu chưa train model nào đó (chưa có file trong checkpoints/), cảnh báo và dùng
    # trọng số khởi tạo ngẫu nhiên — kết quả cột đó trong CSV sẽ không có ý nghĩa so sánh.
    model_specs = {
        'OT_Fusion (Ours)': ('ot_fusion', OTFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4, beta=1.0)),
        'GRACE (beta=0)': ('grace', GRACEFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)),
        'Concat': ('concat', ConcatFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)),
        'Cross-Attention': ('cross_attn', CrossAttentionFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)),
    }

    models = {}
    for display_name, (ckpt_name, model) in model_specs.items():
        model = model.to(device)
        if checkpoint_exists(ckpt_name, save_dir=checkpoint_dir):
            model = load_model(model, ckpt_name, save_dir=checkpoint_dir, device=device)
            print(f"[{display_name}] Đã load checkpoint: {checkpoint_dir}/{ckpt_name}.pt")
        else:
            print(f"[{display_name}] CẢNH BÁO: chưa có checkpoint '{ckpt_name}.pt' — "
                  f"chạy 'python train.py --model {ckpt_name}' trước. Dùng tạm trọng số random.")
        model.eval()
        models[display_name] = model
    
    # Các mức độ desync muốn test (Shift N frames)
    # 0 = đồng bộ, dương = visual trễ, âm = visual sớm
    desync_levels = [-30, -20, -10, 0, 10, 20, 30] 
    
    # Mở file CSV để ghi kết quả
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/robustness_curve.csv'
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Ghi header
        header = ['Desync_Frames'] + list(models.keys())
        writer.writerow(header)
        
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
