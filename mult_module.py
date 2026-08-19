"""
MulT — Multimodal Transformer for Unaligned Multimodal Language Sequences
Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., Salakhutdinov, R.
ACL 2019. arXiv:1906.00295 | https://aclanthology.org/P19-1656/
Official code: https://github.com/yaohungt/Multimodal-Transformer (MIT License,
Copyright (c) 2020 Yao-Hung Hubert Tsai and Shaojie Bai)

`SinusoidalPositionalEmbedding`, `MultiheadAttention`, `TransformerEncoderLayer`
and `TransformerEncoder` below are a direct port of the official
`modules/position_embedding.py`, `modules/multihead_attention.py` and
`modules/transformer.py` (only the module layout was flattened into this one
file; the fairseq-derived attention math is untouched).

`MulTFusionModel` adapts the official `src/models.py::MULTModel` — which fuses
three modalities (language, audio, visual) — down to the **bimodal** case used
in this project (audio, visual): each modality gets ONE directional
crossmodal-attention block attending to the other modality (instead of two,
one per remaining modality), followed by a self-attention "memory" transformer,
then the same residual-projection classifier head as the original.
"""
import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (ported from modules/position_embedding.py)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    """Produces sinusoidal positional embeddings of any length (no padding
    handling needed here since this project always feeds dense, pre-padded
    batches with an explicit mask, not padding-index-encoded sequences)."""

    def __init__(self, embedding_dim, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
        self.weights = None

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, seq_len):
        """Returns positional embeddings of shape (seq_len, embedding_dim)."""
        if self.weights is None or seq_len > self.weights.size(0):
            self.weights = self.get_embedding(seq_len, self.embedding_dim)
        weights = self.weights.type_as(self._float_tensor)
        return weights[:seq_len, :]


# ---------------------------------------------------------------------------
# Multi-head attention (ported from modules/multihead_attention.py, adapted
# from fairseq — supports query/key/value coming from different modalities)
# ---------------------------------------------------------------------------
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None):
        """query/key/value: (seq_len, batch, embed_dim)."""
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight[start:end, :]
        bias = self.in_proj_bias[start:end] if self.in_proj_bias is not None else None
        return F.linear(input, weight, bias)


def buffered_future_mask(dim1, dim2, device):
    future_mask = torch.triu(torch.full((dim1, dim2), float('-inf'), device=device), 1 + abs(dim2 - dim1))
    return future_mask


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


# ---------------------------------------------------------------------------
# Transformer encoder / crossmodal encoder (ported from modules/transformer.py)
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """x: (seq_len, batch, embed_dim). If x_k/x_v given, does crossmodal
        attention (query=x, key=value=x_k/x_v); otherwise self-attention."""
        residual = x
        x = self.layer_norms[0](x)
        mask = buffered_future_mask(x.size(0), (x_k if x_k is not None else x).size(0), x.device) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.layer_norms[0](x_k)
            x_v = self.layer_norms[0](x_v)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    """Stack of `layers` TransformerEncoderLayer, with sinusoidal position
    embeddings added to the input(s). Used both for crossmodal attention
    (x_in_k/x_in_v given) and for the self-attention "memory" transformer
    (x_in_k/x_in_v omitted)."""

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout,
                                     relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """x_in[_k/_v]: (seq_len, batch, embed_dim). Returns (seq_len, batch, embed_dim)."""
        x = self.embed_scale * x_in
        x = x + self.embed_positions(x_in.size(0)).unsqueeze(1).type_as(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k + self.embed_positions(x_in_k.size(0)).unsqueeze(1).type_as(x_in_k)
            x_v = self.embed_scale * x_in_v + self.embed_positions(x_in_v.size(0)).unsqueeze(1).type_as(x_in_v)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)

        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Bimodal MulT fusion model (adapted from src/models.py::MULTModel)
# ---------------------------------------------------------------------------
class MulTFusionModel(nn.Module):
    """
    Bimodal (audio + visual) MulT: each modality attends to the other via a
    directional crossmodal Transformer, then passes through a self-attention
    "memory" Transformer; the two resulting last-timestep representations are
    concatenated and fed through MulT's residual projection head.

    This is the paper's actual crossmodal-attention mechanism (not a single
    nn.MultiheadAttention layer) — used here as the `cross_attn` baseline.
    """

    def __init__(self, audio_dim, visual_dim, hidden_dim, num_classes,
                 num_heads=4, layers=4, mem_layers=3,
                 attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 embed_dropout=0.1, out_dropout=0.1, attn_mask=False):
        super().__init__()
        d = hidden_dim  # common crossmodal embedding dim per modality
        self.d = d
        self.out_dropout = out_dropout

        # 1. Temporal convolutional projections to a common dimension
        self.proj_a = nn.Conv1d(audio_dim, d, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(visual_dim, d, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal attention: each modality attends to the other
        self.trans_a_with_v = TransformerEncoder(d, num_heads, layers, attn_dropout,
                                                   relu_dropout, res_dropout, embed_dropout, attn_mask)
        self.trans_v_with_a = TransformerEncoder(d, num_heads, layers, attn_dropout,
                                                   relu_dropout, res_dropout, embed_dropout, attn_mask)

        # 3. Self-attention "memory" transformer on top of each crossmodal output
        self.trans_a_mem = TransformerEncoder(d, num_heads, max(layers, mem_layers), attn_dropout,
                                                relu_dropout, res_dropout, embed_dropout, attn_mask)
        self.trans_v_mem = TransformerEncoder(d, num_heads, max(layers, mem_layers), attn_dropout,
                                                relu_dropout, res_dropout, embed_dropout, attn_mask)

        # 4. Residual projection + classifier head (as in the official MulT code)
        combined_dim = 2 * d
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, num_classes)

    def forward(self, audio, visual):
        """
        Args:
            audio (Tensor): (B, seq_len_a, audio_dim)
            visual (Tensor): (B, seq_len_v, visual_dim)
        """
        # Conv1d expects (B, channels, seq_len)
        x_a = self.proj_a(audio.transpose(1, 2)).permute(2, 0, 1)   # (seq_len_a, B, d)
        x_v = self.proj_v(visual.transpose(1, 2)).permute(2, 0, 1)  # (seq_len_v, B, d)

        # (V) --> A , (A) --> V
        h_a_with_v = self.trans_a_with_v(x_a, x_v, x_v)
        h_v_with_a = self.trans_v_with_a(x_v, x_a, x_a)

        h_a = self.trans_a_mem(h_a_with_v)
        h_v = self.trans_v_mem(h_v_with_a)

        last_h_a = h_a[-1]  # (B, d) — last timestep, as in the official code
        last_h_v = h_v[-1]

        last_hs = torch.cat([last_h_a, last_h_v], dim=1)  # (B, 2d)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj = last_hs_proj + last_hs

        return self.out_layer(last_hs_proj)
