"""
Tiện ích train/checkpoint — phỏng theo src/utils.py của MulT gốc
(https://github.com/yaohungt/Multimodal-Transformer), rút gọn cho pipeline
của project này (checkpoint theo state_dict thay vì pickle cả model).
"""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Cố định seed cho torch/numpy/random để kết quả reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, model_name, save_dir='checkpoints'):
    """Lưu state_dict của model tốt nhất (theo validation loss) ra checkpoints/<model_name>.pt"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{model_name}.pt')
    torch.save(model.state_dict(), path)
    return path


def load_model(model, model_name, save_dir='checkpoints', device='cpu'):
    """Load state_dict từ checkpoints/<model_name>.pt vào model (in-place) rồi trả về model."""
    path = os.path.join(save_dir, f'{model_name}.pt')
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def checkpoint_exists(model_name, save_dir='checkpoints'):
    return os.path.exists(os.path.join(save_dir, f'{model_name}.pt'))
