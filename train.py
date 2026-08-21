import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from dataset import AudioVisualDataset, collate_fn, compute_class_weights, simulate_desync
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
    # Dữ liệu: đọc feature đã trích sẵn theo docs/FEATURE_CONTRACT.md
    parser.add_argument('--data_root', type=str, required=True,
                        help="Thư mục feature (audio/, visual/, labels.csv, meta.json) — xem docs/FEATURE_CONTRACT.md")
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Fold dùng làm tập TEST theo protocol 5-fold; train trên 4 fold còn lại")
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="Tỉ lệ cắt từ tập train ra làm validation (để chọn checkpoint mà không đụng test set)")
    parser.add_argument('--val_seed', type=int, default=20260101,
                        help="Seed cắt val, chỉ phụ thuộc fold — ĐỪNG đổi giữa 4 model, cả 4 phải dùng chung 1 val split")
    parser.add_argument('--class_weighted', action='store_true',
                        help="Dùng trọng số nghịch tần suất lớp cho CrossEntropyLoss (nên bật với DFEW/MAFW vì lệch lớp nặng)")
    parser.add_argument('--num_workers', type=int, default=2, help="Số worker của DataLoader")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
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

    # Khởi tạo dataset — train = 4 fold còn lại, test = fold được chỉ định.
    # Protocol 5-fold gốc không có tập validation riêng, nên cắt val_ratio từ tập
    # train để chọn checkpoint; test set không bao giờ được dùng để chọn model.
    full_train = AudioVisualDataset(args.data_root, fold=args.fold, split='train')
    test_dataset = AudioVisualDataset(args.data_root, fold=args.fold, split='test')

    # Seed cắt val CHỈ phụ thuộc (dataset, fold) — cố tình không dùng args.seed và
    # không dính dáng tới args.model, để cả 4 model trong cùng 1 fold nhìn thấy y
    # hệt một val split. Nếu mỗi model tự random val riêng thì chênh lệch accuracy
    # giữa chúng lẫn cả nhiễu do val khác nhau, không còn là so sánh công bằng.
    n_val = int(len(full_train) * args.val_ratio)
    train_dataset, val_dataset = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.val_seed + args.fold))
    print(f"Split: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
          f"(val split seed={args.val_seed + args.fold}, dùng chung cho cả 4 model)")

    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Kích thước feature và số lớp lấy từ meta.json, không hardcode
    dims = dict(audio_dim=full_train.audio_dim, visual_dim=full_train.visual_dim,
                hidden_dim=args.hidden_dim, num_classes=full_train.num_classes)

    # Khởi tạo model
    if args.model == 'ot_fusion':
        model = OTFusionModel(**dims, beta=args.beta)
    elif args.model == 'grace':
        model = GRACEFusionModel(**dims)
    elif args.model == 'concat':
        model = ConcatFusionModel(**dims)
    elif args.model == 'cross_attn':
        model = CrossAttentionFusionModel(**dims)

    model = model.to(device)

    class_weights = None
    if args.class_weighted:
        class_weights = compute_class_weights(args.data_root, fold=args.fold, split='train').to(device)
        print("Trọng số lớp (nghịch tần suất trên tập train): "
              + ', '.join(f"{n}={w:.2f}" for n, w in zip(full_train.class_names, class_weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    # Decay LR khi val loss không cải thiện — như scheduler của MulT gốc (ReduceLROnPlateau, factor=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.1)

    # Tên checkpoint kèm fold để chạy 5 fold không ghi đè lẫn nhau
    run_name = f"{args.model}_fold{args.fold}"
    print(f"Bắt đầu huấn luyện mô hình: {args.model} (fold {args.fold}) -> {run_name}.pt")
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
            path = save_model(model, run_name, save_dir=args.checkpoint_dir)
            print(f"  -> Đã lưu checkpoint tốt nhất: {path}")

    # Đánh giá cuối cùng trên test set bằng checkpoint tốt nhất (nếu có lưu)
    if not args.no_save:
        model = load_model(model, run_name, save_dir=args.checkpoint_dir, device=device)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desync_frames=args.desync_frames)
    print(f"\n[Test - best checkpoint] Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
