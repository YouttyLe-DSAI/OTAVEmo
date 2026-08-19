import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import AudioVisualDataset, collate_fn, simulate_desync
from models import OTFusionModel, GRACEFusionModel, ConcatFusionModel, CrossAttentionFusionModel
from utils import set_seed, save_model, load_model
import argparse

def train_epoch(model, dataloader, criterion, optimizer, device, desync_frames=0, clip=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        labels = batch['label'].to(device)

        # Áp dụng desync nếu cần
        if desync_frames != 0:
            visual = simulate_desync(visual, desync_frames)

        optimizer.zero_grad()
        outputs = model(audio, visual)
        loss = criterion(outputs, labels)

        loss.backward()
        # Gradient clipping (như pipeline gốc của MulT, tránh exploding gradient qua Sinkhorn/attention)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return total_loss / len(dataloader), acc

def eval_epoch(model, dataloader, criterion, device, desync_frames=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            labels = batch['label'].to(device)
            
            # Áp dụng desync khi test độ bền (robustness)
            if desync_frames != 0:
                visual = simulate_desync(visual, desync_frames)
                
            outputs = model(audio, visual)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    return total_loss / len(dataloader), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ot_fusion', choices=['ot_fusion', 'grace', 'concat', 'cross_attn'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--desync_frames', type=int, default=0, help="Số frame bị lệch khi train/test")
    parser.add_argument('--beta', type=float, default=1.0, help="Trọng số temporal distance cho OT")
    # Các hyperparameter pipeline (phỏng theo main.py/train.py gốc của MulT)
    parser.add_argument('--optim', type=str, default='Adam', help="Optimizer (Adam, SGD, ...)")
    parser.add_argument('--clip', type=float, default=0.8, help="Gradient clipping norm (0 = tắt)")
    parser.add_argument('--lr_patience', type=int, default=5, help="Số epoch val loss không giảm trước khi decay LR (ReduceLROnPlateau)")
    parser.add_argument('--seed', type=int, default=1111, help="Random seed cho reproducibility")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Thư mục lưu checkpoint tốt nhất")
    parser.add_argument('--no_save', action='store_true', help="Không lưu checkpoint (chỉ để debug nhanh)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Khởi tạo dataset
    train_dataset = AudioVisualDataset(split='train', num_samples=1000)
    val_dataset = AudioVisualDataset(split='val', num_samples=200)
    test_dataset = AudioVisualDataset(split='test', num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model
    if args.model == 'ot_fusion':
        model = OTFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4, beta=args.beta)
    elif args.model == 'grace':
        model = GRACEFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)
    elif args.model == 'concat':
        model = ConcatFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)
    elif args.model == 'cross_attn':
        model = CrossAttentionFusionModel(audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    # Decay LR khi val loss không cải thiện — như scheduler của MulT gốc (ReduceLROnPlateau, factor=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.1)

    print(f"Bắt đầu huấn luyện mô hình: {args.model}")
    best_val_loss = float('inf')
    clip = args.clip if args.clip > 0 else None
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                             desync_frames=args.desync_frames, clip=clip)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desync_frames=args.desync_frames)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Checkpoint model tốt nhất theo validation loss (để run_ablation.py load lại)
        if not args.no_save and val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_model(model, args.model, save_dir=args.checkpoint_dir)
            print(f"  -> Đã lưu checkpoint tốt nhất: {path}")

    # Đánh giá cuối cùng trên test set bằng checkpoint tốt nhất (nếu có lưu)
    if not args.no_save:
        model = load_model(model, args.model, save_dir=args.checkpoint_dir, device=device)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desync_frames=args.desync_frames)
    print(f"\n[Test - best checkpoint] Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
