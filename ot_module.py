import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_ZERO = -1e6   # thay cho log(0), đủ âm để exp() thành 0 mà vẫn hữu hạn


def sinkhorn_knopp(cost_matrix, a=None, b=None, eps=0.1, max_iter=100, tol=1e-7):
    """
    Sinkhorn ở LOG-DOMAIN để giải Optimal Transport có entropy regularization.

    Args:
        cost_matrix (Tensor): Ma trận chi phí, shape (B, N, M)
        a (Tensor|None): Marginal nguồn, shape (B, N), mỗi hàng tổng = 1.
                         Cho phép bằng 0 ở vị trí padding. None -> đều trên N.
        b (Tensor|None): Marginal đích, shape (B, M). None -> đều trên M.
        eps (float): Hệ số entropy regularization
        max_iter (int): Số vòng lặp tối đa
        tol (float): Ngưỡng dừng, đo trên vi phạm ràng buộc biên

    Returns:
        T (Tensor): Transport plan, shape (B, N, M)

    Vì sao log-domain: bản nhân trực tiếp tính K = exp(-C/eps) rồi lặp
    u = a / (K v). Với eps nhỏ, K xuống cỡ 1e-9 hoặc nhỏ hơn, nên hằng số
    ổn định `+1e-8` ở mẫu số LỚN HƠN chính giá trị thật và nuốt mất kết quả —
    đo được sai số ràng buộc biên đọng ở 1e-3 và transport plan phụ thuộc cả
    vào lượng padding. Ở log-domain mọi phép tính đi qua logsumexp nên
    eps nhỏ tới 0.005 vẫn không tràn số.

    Marginal bằng 0 (padding) được mã hoá bằng log-mass rất âm, khiến hàng/cột
    tương ứng của T bằng đúng 0.
    """
    B, N, M = cost_matrix.shape
    dtype, device = cost_matrix.dtype, cost_matrix.device

    if a is None:
        a = torch.full((B, N), 1.0 / N, dtype=dtype, device=device)
    if b is None:
        b = torch.full((B, M), 1.0 / M, dtype=dtype, device=device)

    log_a = torch.where(a > 0, torch.log(a.clamp_min(1e-30)),
                        torch.full_like(a, _LOG_ZERO))
    log_b = torch.where(b > 0, torch.log(b.clamp_min(1e-30)),
                        torch.full_like(b, _LOG_ZERO))

    # Thế năng đối ngẫu f, g (đơn vị chi phí). log T_ij = (f_i + g_j - C_ij) / eps
    f = torch.zeros(B, N, dtype=dtype, device=device)
    g = torch.zeros(B, M, dtype=dtype, device=device)

    for _ in range(max_iter):
        f = eps * (log_a - torch.logsumexp((g.unsqueeze(1) - cost_matrix) / eps, dim=2))
        g = eps * (log_b - torch.logsumexp((f.unsqueeze(2) - cost_matrix) / eps, dim=1))

        log_T = (f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) / eps
        row = torch.logsumexp(log_T, dim=2).exp()
        if torch.max(torch.abs(row - a)) < tol:
            break

    return ((f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) / eps).exp()


def _lengths_from_mask(mask, size, batch, device):
    """mask (B, T) bool -> độ dài thật (B,) float. mask=None -> full length."""
    if mask is None:
        return torch.full((batch,), float(size), device=device)
    return mask.sum(dim=1).clamp(min=1).to(torch.float32)


class OTTemporalAlign(nn.Module):
    """
    Module OT Temporal Alignment.
    Kết hợp khoảng cách ngữ nghĩa (Feature Similarity) và khoảng cách thời gian
    (Temporal Distance).

    Hỗ trợ mask padding: độ dài thật của từng mẫu được dùng cho CẢ marginal của
    Sinkhorn LẪN việc chuẩn hoá chỉ số thời gian. Không có mask, một clip 20 frame
    nằm chung batch với clip 800 frame sẽ bị coi như dài 800 -> temporal_dist sai
    hoàn toàn và vùng padding vẫn tham gia transport.
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

    def forward(self, feat_a, feat_v, mask_a=None, mask_v=None):
        """
        Args:
            feat_a (Tensor): Đặc trưng Audio, shape (B, N, D)
            feat_v (Tensor): Đặc trưng Visual, shape (B, M, D)
            mask_a (Tensor|None): (B, N) bool, True ở vị trí có dữ liệu thật
            mask_v (Tensor|None): (B, M) bool

        Returns:
            aligned_feat_v (Tensor): Visual đã align theo trục thời gian của Audio, (B, N, D)
            T (Tensor): Transport plan, (B, N, M)
        """
        B, N, D = feat_a.shape
        _, M, _ = feat_v.shape
        device, dtype = feat_a.device, feat_a.dtype

        len_a = _lengths_from_mask(mask_a, N, B, device).to(dtype)   # (B,)
        len_v = _lengths_from_mask(mask_v, M, B, device).to(dtype)

        # 1. Feature distance (cosine distance)
        feat_a_norm = F.normalize(feat_a, p=2, dim=-1)
        feat_v_norm = F.normalize(feat_v, p=2, dim=-1)
        feat_dist = 1.0 - torch.bmm(feat_a_norm, feat_v_norm.transpose(1, 2))  # (B, N, M)

        # 2. Temporal distance — chuẩn hoá theo ĐỘ DÀI THẬT của từng mẫu
        pos_a = torch.arange(N, dtype=dtype, device=device).view(1, N) / len_a.view(B, 1)
        pos_v = torch.arange(M, dtype=dtype, device=device).view(1, M) / len_v.view(B, 1)
        temporal_dist = torch.abs(pos_a.unsqueeze(2) - pos_v.unsqueeze(1))     # (B, N, M)

        # 3. Total cost
        cost_matrix = feat_dist + self.beta * temporal_dist

        # 4. Marginal đều trên phần HỢP LỆ, bằng 0 ở padding
        if mask_a is None:
            a = torch.full((B, N), 1.0, dtype=dtype, device=device) / N
        else:
            a = mask_a.to(dtype) / len_a.view(B, 1)
        if mask_v is None:
            b = torch.full((B, M), 1.0, dtype=dtype, device=device) / M
        else:
            b = mask_v.to(dtype) / len_v.view(B, 1)

        # 5. Transport plan
        T = sinkhorn_knopp(cost_matrix, a=a, b=b, eps=self.eps, max_iter=self.max_iter)

        # 6. Kéo Visual về trục thời gian của Audio (chuẩn hoá theo hàng)
        T_normalized = T / (T.sum(dim=2, keepdim=True) + 1e-8)
        aligned_feat_v = torch.bmm(T_normalized, feat_v)

        if mask_a is not None:
            aligned_feat_v = aligned_feat_v * mask_a.unsqueeze(-1).to(dtype)

        return aligned_feat_v, T
