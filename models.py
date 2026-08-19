import torch
import torch.nn as nn
from ot_module import OTTemporalAlign
from mult_module import MulTFusionModel

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


class CrossAttentionFusionModel(MulTFusionModel):
    """
    Baseline Cross-Attention: MulT — Multimodal Transformer for Unaligned
    Multimodal Language Sequences (Tsai et al., ACL 2019, arXiv:1906.00295).
    Ported to the bimodal (audio + visual) case in `mult_module.py` — see đó
    để biết chi tiết kiến trúc (crossmodal attention 2 chiều + self-attention
    "memory" transformer + residual classifier head, đúng theo code gốc).
    """
    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes,
                 num_heads=4, layers=4, mem_layers=3, attn_dropout=0.1,
                 relu_dropout=0.1, res_dropout=0.1, embed_dropout=0.1,
                 out_dropout=0.1, attn_mask=False):
        super().__init__(audio_dim, visual_dim, hidden_dim, num_classes,
                          num_heads=num_heads, layers=layers, mem_layers=mem_layers,
                          attn_dropout=attn_dropout, relu_dropout=relu_dropout,
                          res_dropout=res_dropout, embed_dropout=embed_dropout,
                          out_dropout=out_dropout, attn_mask=attn_mask)
