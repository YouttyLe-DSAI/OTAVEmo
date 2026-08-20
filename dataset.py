import torch
from torch.utils.data import Dataset
import numpy as np

def simulate_desync(visual_features, shift_frames=0, mask=None):
    """
    Giả lập desynchronization bằng cách shift visual features theo chiều thời gian.

    Args:
        visual_features (Tensor): shape (seq_len, feature_dim) HOẶC batch
                                   (batch_size, seq_len, feature_dim) — hàm tự
                                   nhận diện qua số chiều (ndim) của tensor.
        shift_frames (int): Số lượng frame muốn shift.
                            Nếu > 0: visual đi chậm hơn audio (shift phải, padding trái).
                            Nếu < 0: visual đi nhanh hơn audio (shift trái, padding phải).
        mask (Tensor|None): (..., seq_len) bool, True = vị trí có dữ liệu thật.

    Returns:
        shifted_visual (Tensor) nếu mask=None,
        ngược lại (shifted_visual, shifted_mask).

    Mask PHẢI được shift cùng, nếu không vùng padding mới sinh ra sẽ bị coi là
    dữ liệu hợp lệ còn phần dữ liệu bị đẩy ra sẽ bị coi là padding — đúng bằng
    `|shift_frames|` ô sai ở mỗi đầu.
    """
    if shift_frames == 0:
        return visual_features if mask is None else (visual_features, mask)

    seq_len = visual_features.shape[-2]
    shifted = torch.zeros_like(visual_features)
    shifted_mask = torch.zeros_like(mask) if mask is not None else None

    if shift_frames > 0:
        # Visual chậm hơn: các frame từ 0 đến seq_len-shift_frames sẽ được đẩy sang phải
        if shift_frames < seq_len:
            shifted[..., shift_frames:, :] = visual_features[..., :-shift_frames, :]
            if shifted_mask is not None:
                shifted_mask[..., shift_frames:] = mask[..., :-shift_frames]
        # Phần trống bên trái mặc định là zeros (có thể thay đổi padding strategy nếu cần)
    else:
        # Visual nhanh hơn: các frame từ -shift_frames đến cuối sẽ được đẩy sang trái
        shift = abs(shift_frames)
        if shift < seq_len:
            shifted[..., :-shift, :] = visual_features[..., shift:, :]
            if shifted_mask is not None:
                shifted_mask[..., :-shift] = mask[..., shift:]

    if mask is None:
        return shifted
    # Mọi mẫu phải còn ít nhất 1 bước hợp lệ, nếu không Sinkhorn/pooling chia cho 0
    empty = ~shifted_mask.any(dim=-1)
    if empty.any():
        shifted_mask[empty, 0] = True
    return shifted, shifted_mask

class AudioVisualDataset(Dataset):
    """
    Dataset giả lập cho IEMOCAP hoặc CMU-MOSEI đã trích xuất đặc trưng.
    Thực tế bạn sẽ thay hàm __init__ để đọc từ pre-extracted pickle/hdf5/npy.
    """
    def __init__(self, data_path=None, split='train', max_seq_len=50, audio_dim=768, visual_dim=256, num_samples=1000):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.num_samples = num_samples
        
        # TODO: Thay thế phần dưới đây bằng logic đọc dữ liệu thực
        print(f"[{split}] Initializing dummy dataset (audio_dim={audio_dim}, visual_dim={visual_dim})")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # TODO: Lấy dữ liệu thực từ disk theo idx
        
        # Giả lập data ngẫu nhiên
        seq_len_a = np.random.randint(self.max_seq_len // 2, self.max_seq_len)
        seq_len_v = seq_len_a # Trong thực tế có thể khác nhau đôi chút
        
        audio_feat = torch.randn(seq_len_a, self.audio_dim)
        visual_feat = torch.randn(seq_len_v, self.visual_dim)
        
        # Nhãn giả định (classification)
        label = torch.randint(0, 4, (1,)).item() # 4 class cảm xúc
        
        return {
            'audio': audio_feat,
            'visual': visual_feat,
            'label': label,
            'seq_len_a': seq_len_a,
            'seq_len_v': seq_len_v
        }

def collate_fn(batch):
    """
    Hàm gom batch cho dataloader, pad các sequence cho bằng nhau
    """
    audio_list = [item['audio'] for item in batch]
    visual_list = [item['visual'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    visual_padded = torch.nn.utils.rnn.pad_sequence(visual_list, batch_first=True)
    
    # Lấy mask (True ở những vị trí có data)
    audio_mask = (torch.arange(audio_padded.size(1))[None, :] < torch.tensor([item['seq_len_a'] for item in batch])[:, None])
    visual_mask = (torch.arange(visual_padded.size(1))[None, :] < torch.tensor([item['seq_len_v'] for item in batch])[:, None])
    
    return {
        'audio': audio_padded,
        'visual': visual_padded,
        'audio_mask': audio_mask,
        'visual_mask': visual_mask,
        'label': labels
    }
