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


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left-pad only to preserve causality
        x_pad = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x_pad)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.drop(h)
        # conv1 uses causal padding so time dimension already matches original input
        return h + self.residual(x)


class TCNStack(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, layers: int, kernel_size: int = 3, dilation: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        c = in_ch
        for _ in range(layers):
            blocks.append(TemporalConvBlock(c, hid_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            c = hid_ch
        self.net = nn.Sequential(*blocks)

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
        short_layers: int = 2,
        mid_layers: int = 4,
        long_layers: int = 6,
    ):
        super().__init__()
        self.enhancer = FeatureEnhancer(fft_topk=fft_topk, fft_channel_idx=fft_channel_idx)
        enhanced_in = in_feat + max(0, fft_topk)
        # multi-scale TCN experts with fixed kernel/dilation (k=3, dilation=2) and varying depth
        self.tcn_short = TCNStack(enhanced_in, hidden, layers=short_layers, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn_mid = TCNStack(enhanced_in, hidden, layers=mid_layers, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn_long = TCNStack(enhanced_in, hidden, layers=long_layers, kernel_size=3, dilation=2, dropout=dropout)
        self.attn_short = TemporalAttention(hidden)
        self.attn_mid = TemporalAttention(hidden)
        self.attn_long = TemporalAttention(hidden)
        self.fusion = ExpertFusionAttention(hidden_dim=hidden, num_heads=1, use_mha=True)

    def forward(self, x_seq: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: [B, C, T]; cond reserved for potential conditioning
        x_enh = self.enhancer(x_seq)
        h_s = self.tcn_short(x_enh)
        h_m = self.tcn_mid(x_enh)
        h_l = self.tcn_long(x_enh)
        c_s = self.attn_short(h_s)
        c_m = self.attn_mid(h_m)
        c_l = self.attn_long(h_l)
        contexts = torch.stack([c_s, c_m, c_l], dim=1)
        fused = self.fusion(contexts)
        return fused

