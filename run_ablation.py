import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import os

from dataset import AudioVisualDataset, collate_fn
from models import OTFusionModel, GRACEFusionModel, ConcatFusionModel, CrossAttentionFusionModel
from train import eval_epoch

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu chạy Ablation Study trên: {device}")
    
    # 1. Chuẩn bị dataset (Chỉ cần validation/test set cho phần ablation)
    val_dataset = AudioVisualDataset(split='test', num_samples=500)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Khởi tạo các pre-trained models (Giả định đã load checkpoint)
    # Thực tế bạn cần thêm logic load state_dict tại đây
    models = {
        'OT_Fusion (Ours)': OTFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4, beta=1.0).to(device),
        'GRACE (beta=0)': GRACEFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4).to(device),
        'Concat': ConcatFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4).to(device),
        'Cross-Attention': CrossAttentionFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4).to(device)
    }
    
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
