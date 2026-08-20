import torch
import torch.nn as nn
from ot_module import OTTemporalAlign
from mult_module import MulTFusionModel

MODEL_KEYS = ('ot_fusion', 'grace', 'concat', 'cross_attn')


def count_params(model):
    """Số tham số học được — báo cáo trong bảng kết quả để chứng minh so sánh công bằng."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(key, audio_dim=768, visual_dim=256, hidden_dim=128, num_classes=4,
                beta=1.0, eps=0.1, mult_hidden_dim=64, mult_layers=3):
    """Nơi DUY NHẤT khởi tạo model, để train.py và run_ablation.py không bao giờ
    lệch cấu hình nhau (checkpoint load vào kiến trúc khác là lỗi âm thầm).

    MulT dùng `mult_hidden_dim` riêng nhằm cân số tham số — xem docstring của
    CrossAttentionFusionModel.
    """
    common = dict(audio_dim=audio_dim, visual_dim=visual_dim, num_classes=num_classes)
    if key == 'ot_fusion':
        return OTFusionModel(hidden_dim=hidden_dim, eps=eps, beta=beta, **common)
    if key == 'grace':
        return GRACEFusionModel(hidden_dim=hidden_dim, eps=eps, **common)
    if key == 'concat':
        return ConcatFusionModel(hidden_dim=hidden_dim, **common)
    if key == 'cross_attn':
        return CrossAttentionFusionModel(hidden_dim=mult_hidden_dim,
                                         layers=mult_layers, mem_layers=mult_layers,
                                         **common)
    raise ValueError(f"model không hợp lệ: {key!r} (hợp lệ: {MODEL_KEYS})")


def masked_mean(x, mask):
    """x: (B, T, D), mask: (B, T) bool True=hợp lệ -> (B, D).
    Mean pooling thường sẽ kéo trung bình về 0 vì đếm cả vùng padding."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class FeatureEncoder(nn.Module):
    """
    Encoder đơn giản để chuẩn hóa feature dimension trước khi fusion.
    Bạn có thể đổi sang TransformerEncoder nếu cần.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x, mask=None):
        # x: (B, seq_len, input_dim)
        if mask is None:
            out, _ = self.lstm(x)
            return out
        # Không pack thì LSTM đọc cả vùng padding và trạng thái ẩn bị nhiễm
        lengths = mask.to(torch.long).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        return out

class OTFusionModel(nn.Module):
    """
    Mô hình đề xuất sử dụng OT Temporal Alignment.
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes, eps=0.1, beta=1.0):
        super().__init__()
        self.audio_enc = FeatureEncoder(audio_dim, hidden_dim)
        self.visual_enc = FeatureEncoder(visual_dim, hidden_dim)
        
        # OT Module (Temporal-Aware)
        self.ot_align = OTTemporalAlign(eps=eps, beta=beta)
        
        # Classifier sau fusion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, audio, visual, audio_mask=None, visual_mask=None):
        """
        Args:
            audio (Tensor): (B, seq_len_a, audio_dim)
            visual (Tensor): (B, seq_len_v, visual_dim)
            audio_mask (Tensor|None): (B, seq_len_a) bool, True = hợp lệ
            visual_mask (Tensor|None): (B, seq_len_v) bool
        """
        # 1. Encode ra cùng hidden_dim
        feat_a = self.audio_enc(audio, audio_mask)    # (B, seq_len_a, hidden_dim)
        feat_v = self.visual_enc(visual, visual_mask)  # (B, seq_len_v, hidden_dim)

        # 2. Căn chỉnh Visual theo Audio bằng OT
        aligned_feat_v, _ = self.ot_align(feat_a, feat_v, audio_mask, visual_mask)

        # 3. Fusion (concat theo chiều feature)
        # aligned_feat_v giờ có cùng seq_len_a với feat_a
        fused = torch.cat([feat_a, aligned_feat_v], dim=-1)  # (B, seq_len_a, hidden_dim * 2)

        # 4. Pooling theo chiều thời gian, bỏ qua padding
        pooled = masked_mean(fused, audio_mask)  # (B, hidden_dim * 2)

        # 5. Classify
        out = self.classifier(pooled)  # (B, num_classes)
        return out


class GRACEFusionModel(OTFusionModel):
    """
    Baseline OT không có Temporal-aware (GRACE-style).
    Đơn giản là set beta = 0 trong OT module.
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes, eps=0.1):
        super().__init__(audio_dim, visual_dim, hidden_dim, num_classes, eps=eps, beta=0.0)


class ConcatFusionModel(nn.Module):
    """
    Baseline Concat Fusion: mean pooling từng nhánh rồi concat.
    Không xử lý alignment gì.
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes):
        super().__init__()
        self.audio_enc = FeatureEncoder(audio_dim, hidden_dim)
        self.visual_enc = FeatureEncoder(visual_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, audio, visual, audio_mask=None, visual_mask=None):
        feat_a = self.audio_enc(audio, audio_mask)
        feat_v = self.visual_enc(visual, visual_mask)

        pooled_a = masked_mean(feat_a, audio_mask)
        pooled_v = masked_mean(feat_v, visual_mask)

        fused = torch.cat([pooled_a, pooled_v], dim=-1)
        out = self.classifier(fused)
        return out


class CrossAttentionFusionModel(MulTFusionModel):
    """
    Baseline Cross-Attention: MulT — Multimodal Transformer for Unaligned
    Multimodal Language Sequences (Tsai et al., ACL 2019, arXiv:1906.00295).
    Ported to the bimodal (audio + visual) case in `mult_module.py` — see đó
    để biết chi tiết kiến trúc (crossmodal attention 2 chiều + self-attention
    "memory" transformer + residual classifier head, đúng theo code gốc).

    ⚠ SO SÁNH CÔNG BẰNG: MulT phình theo d² (attention + FFN 4x) còn BiLSTM thì
    không, nên đặt CÙNG `hidden_dim` cho mọi model KHÔNG có nghĩa là cùng dung
    lượng — ở d=128 MulT nặng gấp 5.5x các model kia. Paper MulT ghi rõ họ
    "control the number of parameters of all models to be approximately the same".
    Mặc định d=64, layers=3 -> ~699K, cân với ~625K của OT/GRACE/Concat ở d=128.
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim=64, num_classes=4,
                 num_heads=4, layers=3, mem_layers=3, attn_dropout=0.1,
                 relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1,
                 out_dropout=0.1, attn_mask=False):
        super().__init__(audio_dim, visual_dim, hidden_dim, num_classes,
                          num_heads=num_heads, layers=layers, mem_layers=mem_layers,
                          attn_dropout=attn_dropout, relu_dropout=relu_dropout,
                          res_dropout=res_dropout, embed_dropout=embed_dropout,
                          out_dropout=out_dropout, attn_mask=attn_mask)
