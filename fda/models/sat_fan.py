from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FeatureEnhancer(nn.Module):
    def __init__(self, fft_topk: int = 8, fft_channel_idx: int = 0, eps: float = 1e-6):
        super().__init__()
        self.fft_topk = fft_topk
        self.fft_channel_idx = fft_channel_idx
        self.eps = eps

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mean_t = x.mean(dim=-1, keepdim=True)
        std_t = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean_t) / (std_t + self.eps)

    def _fft_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> choose one channel, compute rFFT magnitude and take top-k amplitudes
        B, C, T = x.shape
        ch = min(max(0, self.fft_channel_idx), C - 1)
        sig = x[:, ch, :]  # [B, T]
        spec = torch.fft.rfft(sig, dim=-1)  # [B, F]
        mag = spec.abs() / (T + 1e-9)
        # remove DC component
        if mag.size(-1) > 1:
            mag = mag[:, 1:]
        k = min(self.fft_topk, mag.size(-1))
        if k <= 0:
            return x.new_zeros(B, 0, T)
        vals, _ = torch.topk(mag, k=k, dim=-1)
        # broadcast across time to form stationary frequency features
        freq_feats = vals.unsqueeze(-1).expand(B, k, T)  # [B, K, T]
        return freq_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_norm = self._zscore(x)
        freq_feats = self._fft_features(x_norm)
        if freq_feats.size(1) == 0:
            return x_norm
        x_enhanced = torch.cat([x_norm, freq_feats], dim=1)
        return x_enhanced


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.drop(h)
        # match time dimension (simple right-trim to keep causality tendency)
        if h.size(-1) != x.size(-1):
            h = h[..., : x.size(-1)]
        return h + self.res(x)


class TCNStack(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, layers: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        mods = []
        c = in_ch
        dilation = 1
        for _ in range(layers):
            mods.append(TemporalConvBlock(c, hid_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            c = hid_ch
            dilation *= 2
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.net(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        # h_seq: [B, H, T] -> work in [B, T, H]
        h = h_seq.transpose(1, 2)  # [B, T, H]
        scores = self.proj(torch.tanh(h)).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1)  # [B, T]
        context = torch.einsum('bt,bth->bh', weights, h)
        return context  # [B, H]


class ExpertFusionAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1, use_mha: bool = True):
        super().__init__()
        self.use_mha = use_mha
        if use_mha:
            self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [Tq=1, B=1, H]
            self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
        else:
            self.score = nn.Linear(hidden_dim, 1)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        # contexts: [B, 3, H]
        if self.use_mha:
            B = contexts.size(0)
            kv = contexts.transpose(0, 1)  # [S=3, B, H]
            q = self.query_token.repeat(1, B, 1)  # [1, B, H]
            out, _ = self.mha(q, kv, kv)  # [1, B, H]
            return out.squeeze(0)  # [B, H]
        else:
            # simple gating over experts
            scores = self.score(torch.tanh(contexts)).squeeze(-1)  # [B, 3]
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, 3, 1]
            fused = (weights * contexts).sum(dim=1)
            return fused  # [B, H]


class SATFAN(nn.Module):
    def __init__(
        self,
        in_feat: int,
        hidden: int = 128,
        fft_topk: int = 8,
        fft_channel_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enhancer = FeatureEnhancer(fft_topk=fft_topk, fft_channel_idx=fft_channel_idx)
        enhanced_in = in_feat + max(0, fft_topk)
        # multi-scale TCN experts with different receptive fields
        self.tcn_short = TCNStack(enhanced_in, hidden, layers=3, kernel_size=3, dropout=dropout)   # ~15 steps
        self.tcn_mid = TCNStack(enhanced_in, hidden, layers=6, kernel_size=3, dropout=dropout)     # ~127 steps
        self.tcn_long = TCNStack(enhanced_in, hidden, layers=8, kernel_size=3, dropout=dropout)    # ~511 steps
        # temporal attention per expert
        self.attn_short = TemporalAttention(hidden)
        self.attn_mid = TemporalAttention(hidden)
        self.attn_long = TemporalAttention(hidden)
        # expert fusion attention
        self.fusion = ExpertFusionAttention(hidden_dim=hidden, num_heads=1, use_mha=True)

    def forward(self, x_seq: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: [B, C, T]; cond is unused but kept for interface compatibility
        x_enh = self.enhancer(x_seq)
        h_s = self.tcn_short(x_enh)  # [B, H, T]
        h_m = self.tcn_mid(x_enh)
        h_l = self.tcn_long(x_enh)
        c_s = self.attn_short(h_s)   # [B, H]
        c_m = self.attn_mid(h_m)
        c_l = self.attn_long(h_l)
        contexts = torch.stack([c_s, c_m, c_l], dim=1)  # [B, 3, H]
        fused = self.fusion(contexts)  # [B, H]
        return fused

