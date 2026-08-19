# OTAVEmo — Temporal-Aware Optimal Transport for Audio-Visual Emotion Recognition

Nghiên cứu hướng tới paper Rank B: **Temporal-Aware Optimal Transport for Robust
Audio-Visual Fusion under Modality Desynchronization**, target hội nghị **SAC 2027**
(ACM/SIGAPP, hạn nộp 2/10/2026). Xem chi tiết scope/timeline trong [prompt.md](prompt.md).

**Ý tưởng chính:** thêm temporal-distance term vào cost matrix của Optimal Transport
(`beta * |i/N - j/M|`) để việc align audio↔visual bằng Sinkhorn bền vững hơn khi 2
luồng bị lệch pha (desync), so với OT chỉ dùng feature similarity (`beta=0`) và với
crossmodal-attention (MulT).

## Cấu trúc project

| File | Mô tả |
|---|---|
| [ot_module.py](ot_module.py) | `OTTemporalAlign` — module đề xuất chính: Sinkhorn-Knopp trên cost matrix = feature distance + β·temporal distance |
| [mult_module.py](mult_module.py) | Port của **MulT** (Tsai et al., ACL 2019, [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)) từ [source gốc](https://github.com/yaohungt/Multimodal-Transformer) (MIT License), rút gọn từ 3 modality (language/audio/visual) xuống 2 modality (audio/visual). Đã verify khớp số học 1:1 với code gốc (xem lịch sử làm việc) |
| [models.py](models.py) | 4 model so sánh: `OTFusionModel` (đề xuất), `GRACEFusionModel` (OT với β=0), `ConcatFusionModel`, `CrossAttentionFusionModel` (MulT) |
| [dataset.py](dataset.py) | `AudioVisualDataset` (hiện là dữ liệu giả — cần thay bằng loader thật khi có dataset), `simulate_desync` (shift artificial theo N frame) |
| [extract_features.py](extract_features.py) | Skeleton trích feature audio (wav2vec2) / visual (VideoMAE) — chưa cài model thật |
| [train.py](train.py) | Train 1 model: gradient clipping, `ReduceLROnPlateau`, checkpoint theo best val loss, eval trên test set bằng best checkpoint |
| [run_ablation.py](run_ablation.py) | Load checkpoint đã train của cả 4 model, đánh giá qua nhiều mức desync, xuất `results/robustness_curve.csv` |
| [utils.py](utils.py) | `set_seed`, `save_model`/`load_model` (checkpoint theo state_dict) |

## Cài đặt môi trường

```bash
conda create -n ot python=3.10 -y
conda activate ot
pip install -r requirements.txt
```

## Cách chạy

```bash
# Train 1 model (checkpoint tốt nhất tự lưu vào checkpoints/<model>.pt)
python train.py --model ot_fusion --epochs 10 --beta 1.0
python train.py --model grace
python train.py --model concat
python train.py --model cross_attn

# Sau khi đã train đủ 4 model, chạy ablation robustness curve
python run_ablation.py
```

Tham số chính của `train.py`: `--model {ot_fusion,grace,concat,cross_attn}`,
`--epochs`, `--batch_size`, `--lr`, `--beta` (trọng số temporal distance cho OT),
`--desync_frames` (train/eval với desync giả lập), `--clip`, `--lr_patience`,
`--seed`, `--checkpoint_dir`.

## Trạng thái hiện tại

- ✅ Kiến trúc 4 model + pipeline train/checkpoint/ablation đã chạy end-to-end (verify bằng dữ liệu giả).
- ⏳ **Đang chờ dataset thật** (IEMOCAP / CMU-MOSEI) để thay `AudioVisualDataset` giả — xem mục "Bước tiếp theo" trong [prompt.md](prompt.md).
- ⏳ Chưa cài backbone pretrained thật cho `extract_features.py` (wav2vec2 / VideoMAE).

## Trích dẫn baseline

```bibtex
@inproceedings{tsai2019MULT,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Liang, Paul Pu and Kolter, J. Zico and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2019},
  publisher={Association for Computational Linguistics},
}
```
