import csv
import json
import os

import torch
from torch.utils.data import Dataset

# Gợi ý chung cho mọi lỗi liên quan tới format dữ liệu — thay vì để code crash
# mù mờ ở giữa vòng train, chỉ thẳng người dùng sang script kiểm tra.
_CHECK_HINT = ("Chạy `python docs/check_dataset_format.py <dataset_root>` để kiểm tra "
               "thư mục feature trước khi train (đặc tả: docs/FEATURE_CONTRACT.md).")


def simulate_desync(visual_features, shift_frames=0):
    """
    Giả lập desynchronization bằng cách shift visual features theo chiều thời gian.

    Args:
        visual_features (Tensor): shape (seq_len, feature_dim) HOẶC batch
                                   (batch_size, seq_len, feature_dim) — hàm tự
                                   nhận diện qua số chiều (ndim) của tensor.
        shift_frames (int): Số lượng frame muốn shift.
                            Nếu > 0: visual đi chậm hơn audio (shift phải, padding trái).
                            Nếu < 0: visual đi nhanh hơn audio (shift trái, padding phải).

    Returns:
        shifted_visual (Tensor): Tensor cùng shape nhưng đã bị shift theo chiều seq_len.
    """
    if shift_frames == 0:
        return visual_features

    seq_len = visual_features.shape[-2]
    shifted = torch.zeros_like(visual_features)

    if shift_frames > 0:
        # Visual chậm hơn: các frame từ 0 đến seq_len-shift_frames sẽ được đẩy sang phải
        if shift_frames < seq_len:
            shifted[..., shift_frames:, :] = visual_features[..., :-shift_frames, :]
        # Phần trống bên trái mặc định là zeros (có thể thay đổi padding strategy nếu cần)
    else:
        # Visual nhanh hơn: các frame từ -shift_frames đến cuối sẽ được đẩy sang trái
        shift = abs(shift_frames)
        if shift < seq_len:
            shifted[..., :-shift, :] = visual_features[..., shift:, :]

    return shifted


def _load_meta(data_root):
    """Đọc meta.json và kiểm tra các key bắt buộc theo FEATURE_CONTRACT.md mục 3."""
    meta_path = os.path.join(data_root, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Thiếu file bắt buộc: {meta_path}\n{_CHECK_HINT}")

    with open(meta_path, encoding='utf-8') as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{meta_path} không phải JSON hợp lệ: {e}\n{_CHECK_HINT}") from e

    missing = [k for k in ('num_classes', 'class_names', 'audio_dim', 'visual_dim') if k not in meta]
    if missing:
        raise ValueError(f"{meta_path} thiếu key bắt buộc {missing}\n{_CHECK_HINT}")
    if len(meta['class_names']) != meta['num_classes']:
        raise ValueError(
            f"{meta_path}: len(class_names)={len(meta['class_names'])} không khớp "
            f"num_classes={meta['num_classes']}\n{_CHECK_HINT}")
    return meta


def _read_labels(data_root, fold, split, num_classes=None):
    """
    Đọc labels.csv rồi lọc theo fold/split.

    Theo contract, cột `fold` là fold mà clip đó thuộc tập TEST:
      split='test'  -> giữ clip có fold == fold truyền vào
      split='train' -> giữ clip có fold != fold truyền vào

    Returns:
        list các tuple (clip_id, label).
    """
    if split not in ('train', 'test'):
        raise ValueError(f"split phải là 'train' hoặc 'test', nhận được {split!r}")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError(f"fold phải là số nguyên 1-5 (protocol 5-fold), nhận được {fold!r}")

    labels_path = os.path.join(data_root, 'labels.csv')
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Thiếu file bắt buộc: {labels_path}\n{_CHECK_HINT}")

    with open(labels_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [c for c in ('clip_id', 'label', 'fold') if c not in columns]
        if missing:
            raise ValueError(
                f"{labels_path} thiếu cột bắt buộc {missing} (đang có: {columns})\n{_CHECK_HINT}")

        entries, folds_seen = [], set()
        for i, row in enumerate(reader, start=2):  # dòng 1 là header
            clip_id = (row['clip_id'] or '').strip()
            try:
                label, row_fold = int(row['label']), int(row['fold'])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{labels_path} dòng {i}: label/fold không phải số nguyên "
                    f"(label={row['label']!r}, fold={row['fold']!r})\n{_CHECK_HINT}") from e

            if num_classes is not None and not 0 <= label < num_classes:
                raise ValueError(
                    f"{labels_path} dòng {i}: label={label} nằm ngoài [0, {num_classes - 1}] "
                    f"theo meta.json\n{_CHECK_HINT}")

            folds_seen.add(row_fold)
            keep = (row_fold == fold) if split == 'test' else (row_fold != fold)
            if keep:
                entries.append((clip_id, label))

    if not entries:
        raise ValueError(
            f"{labels_path}: không có clip nào cho split='{split}', fold={fold} "
            f"(các fold có trong file: {sorted(folds_seen)})\n{_CHECK_HINT}")
    return entries


class AudioVisualDataset(Dataset):
    """
    Đọc feature đã trích sẵn theo docs/FEATURE_CONTRACT.md:

        <data_root>/audio/<clip_id>.pt    tensor (T_a, D_a) float32
        <data_root>/visual/<clip_id>.pt   tensor (T_v, D_v) float32
        <data_root>/labels.csv            clip_id,label,fold
        <data_root>/meta.json             num_classes, class_names, audio_dim, visual_dim

    Không pad ở đây — mỗi clip giữ nguyên độ dài T thật, việc pad + sinh mask do
    `collate_fn` lo. `num_classes`/`class_names`/dim đọc từ meta.json, không hardcode.
    """

    def __init__(self, data_root, fold, split='train'):
        """
        Args:
            data_root (str): Thư mục dataset (chứa audio/, visual/, labels.csv, meta.json).
            fold (int): Fold 1-5 dùng làm tập TEST theo protocol 5-fold chính thức.
            split (str): 'train' (fold != fold truyền vào) hoặc 'test' (fold == ...).
        """
        super().__init__()
        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dataset: {data_root}\n{_CHECK_HINT}")

        self.data_root = data_root
        self.fold = fold
        self.split = split

        self.audio_dir = os.path.join(data_root, 'audio')
        self.visual_dir = os.path.join(data_root, 'visual')
        for d in (self.audio_dir, self.visual_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Thiếu thư mục bắt buộc: {d}\n{_CHECK_HINT}")

        self.meta = _load_meta(data_root)
        self.num_classes = self.meta['num_classes']
        self.class_names = self.meta['class_names']
        self.audio_dim = self.meta['audio_dim']
        self.visual_dim = self.meta['visual_dim']

        self.entries = _read_labels(data_root, fold, split, num_classes=self.num_classes)

        # Kiểm tra sự tồn tại của file .pt ngay lúc khởi tạo — thà hỏng ở đây còn
        # hơn hỏng ở epoch thứ 3 lúc DataLoader chạm phải clip thiếu file.
        missing = []
        for clip_id, _ in self.entries:
            for path in (self._path('audio', clip_id), self._path('visual', clip_id)):
                if not os.path.isfile(path):
                    missing.append(path)
        if missing:
            shown = '\n  '.join(missing[:5])
            more = f"\n  ... và {len(missing) - 5} file nữa" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"Thiếu {len(missing)} file feature được liệt kê trong labels.csv:\n"
                f"  {shown}{more}\n{_CHECK_HINT}")

        print(f"[{split}] fold={fold}: {len(self.entries)} clip từ {data_root} "
              f"(num_classes={self.num_classes}, audio_dim={self.audio_dim}, "
              f"visual_dim={self.visual_dim})")

    def _path(self, modality, clip_id):
        return os.path.join(self.data_root, modality, f'{clip_id}.pt')

    def _load_feat(self, modality, clip_id, expected_dim):
        path = self._path(modality, clip_id)
        feat = torch.load(path, map_location='cpu')

        if not torch.is_tensor(feat):
            raise TypeError(f"{path}: phải là torch.Tensor, đang là {type(feat).__name__}\n{_CHECK_HINT}")
        if feat.ndim != 2:
            raise ValueError(
                f"{path}: contract yêu cầu tensor 2 chiều (T, D), đang là shape "
                f"{tuple(feat.shape)}\n{_CHECK_HINT}")
        if feat.shape[1] != expected_dim:
            raise ValueError(
                f"{path}: D={feat.shape[1]} không khớp meta.json['{modality}_dim']="
                f"{expected_dim}\n{_CHECK_HINT}")
        return feat.float()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        clip_id, label = self.entries[idx]
        audio_feat = self._load_feat('audio', clip_id, self.audio_dim)
        visual_feat = self._load_feat('visual', clip_id, self.visual_dim)

        return {
            'audio': audio_feat,
            'visual': visual_feat,
            'label': label,
            'seq_len_a': audio_feat.shape[0],
            'seq_len_v': visual_feat.shape[0],
            'clip_id': clip_id,
        }


def compute_class_weights(labels_csv, fold, split='train', num_classes=None):
    """
    Trọng số nghịch tần suất lớp, để truyền vào `nn.CrossEntropyLoss(weight=...)`.

    Cần cho DFEW/MAFW vì phân bố lớp lệch rất nặng (DFEW: happy ~2800 mẫu trong
    khi disgust ~87) — nếu không cân bằng, model dễ bỏ hẳn lớp hiếm mà accuracy
    tổng vẫn đẹp.

    Công thức "balanced" chuẩn: w_c = N / (C * count_c), tức lớp hiếm được đánh
    trọng số cao hơn tỉ lệ nghịch với tần suất, và trung bình có trọng số của w
    xấp xỉ 1 nên độ lớn loss không bị đổi thang.

    Args:
        labels_csv (str): Đường dẫn tới labels.csv (hoặc tới dataset_root chứa nó).
        fold (int): Fold 1-5 dùng làm tập test.
        split (str): Tính trên split nào — mặc định 'train' (KHÔNG dùng 'test',
                     tính trọng số trên tập test là rò rỉ thông tin).
        num_classes (int|None): Nếu None thì đọc từ meta.json cạnh labels.csv.

    Returns:
        Tensor shape (num_classes,), dtype float32. Lớp không có mẫu nào trong
        split này nhận trọng số 0 (thay vì inf).
    """
    if os.path.isdir(labels_csv):
        data_root = labels_csv
    else:
        data_root = os.path.dirname(os.path.abspath(labels_csv))

    if num_classes is None:
        num_classes = _load_meta(data_root)['num_classes']

    entries = _read_labels(data_root, fold, split, num_classes=num_classes)

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in entries:
        counts[label] += 1

    total = counts.sum()
    weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))

    # Lớp trống nhận weight 0 (thay vì inf) — nhưng phải hét lên, vì im lặng ở đây
    # nghĩa là model không bao giờ bị phạt khi bỏ hẳn lớp đó mà log vẫn đẹp.
    empty = (counts == 0).nonzero().flatten().tolist()
    if empty:
        names = _load_meta(data_root)['class_names']
        print(f"CẢNH BÁO: lớp {[names[c] for c in empty]} không có mẫu nào trong split "
              f"'{split}' của fold {fold} -> weight = 0, model sẽ bỏ qua hoàn toàn lớp này. "
              f"Kiểm tra lại phân bố lớp theo fold trước khi lấy số liệu chính thức.")
    return weights


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
