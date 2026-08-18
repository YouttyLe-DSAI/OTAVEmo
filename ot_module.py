import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_knopp(cost_matrix, eps=0.1, max_iter=100, tol=1e-5):
    """
    Thuật toán Sinkhorn để giải bài toán Optimal Transport.
    
    Args:
        cost_matrix (Tensor): Ma trận chi phí, shape (B, N, M)
        eps (float): Hệ số entropy regularization
        max_iter (int): Số vòng lặp tối đa
        tol (float): Ngưỡng dừng
        
    Returns:
        T (Tensor): Transport plan, shape (B, N, M)
    """
    B, N, M = cost_matrix.shape
    
    # Kernel matrix
    K = torch.exp(-cost_matrix / eps)
    
    # Marginal distributions (đồng đều)
    u = torch.ones(B, N, dtype=cost_matrix.dtype, device=cost_matrix.device) / N
    v = torch.ones(B, M, dtype=cost_matrix.dtype, device=cost_matrix.device) / M
    
    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u
        u = 1.0 / (N * torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
        v = 1.0 / (M * torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
        
        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
            
    # Transport plan
    T = torch.bmm(torch.bmm(torch.diag_embed(u), K), torch.diag_embed(v))
    return T

class OTTemporalAlign(nn.Module):
    """
    Module OT Temporal Alignment (Task 3 & 4).
    Kết hợp khoảng cách ngữ nghĩa (Feature Similarity) và khoảng cách thời gian (Temporal Distance).
    """
    def __init__(self, eps=0.1, beta=1.0, max_iter=50):
        """
        Args:
            eps (float): Tham số entropy regularization cho Sinkhorn.
            beta (float): Trọng số của Temporal Distance. 
                          Nếu beta=0 -> GRACE-style OT (chỉ dùng feature similarity).
            max_iter (int): Số vòng lặp của Sinkhorn.
        """
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.max_iter = max_iter

    def forward(self, feat_a, feat_v):
        """
        Args:
            feat_a (Tensor): Đặc trưng Audio, shape (B, N, D)
            feat_v (Tensor): Đặc trưng Visual, shape (B, M, D)
            
        Returns:
            aligned_feat_v (Tensor): Đặc trưng Visual đã được align theo Audio, shape (B, N, D)
            T (Tensor): Transport plan matrix, shape (B, N, M)
        """
        B, N, D = feat_a.shape
        _, M, _ = feat_v.shape
        
        # 1. Feature Distance (Cosine distance / L2 distance)
        # Sử dụng L2 distance (squared) giữa các frame
        feat_a_norm = F.normalize(feat_a, p=2, dim=-1)
        feat_v_norm = F.normalize(feat_v, p=2, dim=-1)
        
        # (B, N, M)
        feat_dist = 1.0 - torch.bmm(feat_a_norm, feat_v_norm.transpose(1, 2)) 
        
        # 2. Temporal Distance
        # Khoảng cách tuyệt đối giữa các chỉ số frame
        idx_a = torch.arange(N, dtype=feat_a.dtype, device=feat_a.device).view(1, N, 1) / N
        idx_v = torch.arange(M, dtype=feat_v.dtype, device=feat_v.device).view(1, 1, M) / M
        temporal_dist = torch.abs(idx_a - idx_v) # (1, N, M)
        temporal_dist = temporal_dist.expand(B, -1, -1)
        
        # 3. Total Cost Matrix
        cost_matrix = feat_dist + self.beta * temporal_dist
        
        # 4. Tính Optimal Transport Plan bằng Sinkhorn
        T = sinkhorn_knopp(cost_matrix, eps=self.eps, max_iter=self.max_iter)
        
        # 5. Căn chỉnh Visual về Audio
        # Nhân T (B, N, M) với feat_v (B, M, D) -> (B, N, D)
        # Scale lại T để tổng các dòng = 1
        T_normalized = T / (T.sum(dim=2, keepdim=True) + 1e-8)
        aligned_feat_v = torch.bmm(T_normalized, feat_v)
        
        return aligned_feat_v, T
