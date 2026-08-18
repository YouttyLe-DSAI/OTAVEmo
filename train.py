import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AudioVisualDataset, collate_fn, simulate_desync
from models import OTFusionModel, GRACEFusionModel, ConcatFusionModel, CrossAttentionFusionModel
import argparse

def train_epoch(model, dataloader, criterion, optimizer, device, desync_frames=0):
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Khởi tạo dataset
    train_dataset = AudioVisualDataset(split='train', num_samples=1000)
    val_dataset = AudioVisualDataset(split='val', num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Bắt đầu huấn luyện mô hình: {args.model}")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, desync_frames=args.desync_frames)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desync_frames=args.desync_frames)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

if __name__ == '__main__':
    main()
