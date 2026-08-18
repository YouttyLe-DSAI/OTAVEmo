import torch
import torch.nn as nn
from ot_module import OTTemporalAlign

class FeatureEncoder(nn.Module):
    """
    Encoder đơn giản để chuẩn hóa feature dimension trước khi fusion.
    Bạn có thể đổi sang TransformerEncoder nếu cần.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
                            
    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
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
        
    def forward(self, audio, visual):
        """
        Args:
            audio (Tensor): (B, seq_len_a, audio_dim)
            visual (Tensor): (B, seq_len_v, visual_dim)
        """
        # 1. Encode ra cùng hidden_dim
        feat_a = self.audio_enc(audio) # (B, seq_len_a, hidden_dim)
        feat_v = self.visual_enc(visual) # (B, seq_len_v, hidden_dim)
        
        # 2. Căn chỉnh Visual theo Audio bằng OT
        aligned_feat_v, _ = self.ot_align(feat_a, feat_v) # (B, seq_len_a, hidden_dim)
        
        # 3. Fusion (ví dụ: concat theo chiều feature)
        # aligned_feat_v giờ có cùng seq_len_a với feat_a
        fused = torch.cat([feat_a, aligned_feat_v], dim=-1) # (B, seq_len_a, hidden_dim * 2)
        
        # 4. Pooling (mean pool theo chiều thời gian)
        pooled = torch.mean(fused, dim=1) # (B, hidden_dim * 2)
        
        # 5. Classify
        out = self.classifier(pooled) # (B, num_classes)
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
        
    def forward(self, audio, visual):
        feat_a = self.audio_enc(audio)
        feat_v = self.visual_enc(visual)
        
        pooled_a = torch.mean(feat_a, dim=1)
        pooled_v = torch.mean(feat_v, dim=1)
        
        fused = torch.cat([pooled_a, pooled_v], dim=-1)
        out = self.classifier(fused)
        return out


class CrossAttentionFusionModel(nn.Module):
    """
    Baseline Cross-Attention (Kiểu MulT).
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes):
        super().__init__()
        self.audio_enc = FeatureEncoder(audio_dim, hidden_dim)
        self.visual_enc = FeatureEncoder(visual_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, audio, visual):
        feat_a = self.audio_enc(audio)
        feat_v = self.visual_enc(visual)
        
        # Query là audio, Key/Value là visual
        attn_out, _ = self.cross_attn(query=feat_a, key=feat_v, value=feat_v)
        
        fused = torch.cat([feat_a, attn_out], dim=-1)
        pooled = torch.mean(fused, dim=1)
        
        out = self.classifier(pooled)
        return out
