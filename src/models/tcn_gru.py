# -*- coding: utf-8 -*-
"""Multi-scale TCN + GRU 骨干，双头：9类 rhythm 分类 + 信号质量回归。

结构（chatGPT建议3 第7-8节）：
  ECG (B,1,L) → 3路多尺度卷积(k=5/9/15) → TCN(dilated residual) → GRU
    → rhythm classifier (9)
    → quality head (sigmoid, 预测 quality_frac ∈ [0,1])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    """3 种尺度的并行卷积捕捉 QRS/P/T 波形态。"""

    def __init__(self, in_ch=1, out_ch=32, kernels=(5, 9, 15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2) for k in kernels])
        self.norm = nn.BatchNorm1d(out_ch * len(kernels))

    def forward(self, x):  # x: (B, 1, L)
        return self.norm(torch.cat([F.relu(c(x)) for c in self.convs], dim=1))


class TCNBlock(nn.Module):
    """膨胀因果残差块（右填充保证因果性）。"""

    def __init__(self, ch, dilation, dropout=0.2):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(ch, ch, 3, dilation=dilation, padding=pad)
        self.norm1 = nn.BatchNorm1d(ch)
        self.norm2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.norm1(self.conv1(x)[:, :, :x.size(2)])))
        h = self.drop(F.relu(self.norm2(self.conv2(h)[:, :, :x.size(2)])))
        return F.relu(x + h)


class TCNGRU(nn.Module):
    def __init__(self, n_class=9, hidden=128, use_quality_head=True):
        super().__init__()
        self.msc = MultiScaleConv(1, 32, (5, 9, 15))
        ch = 96
        self.tcn = nn.Sequential(
            TCNBlock(ch, 1), TCNBlock(ch, 2),
            TCNBlock(ch, 4), TCNBlock(ch, 8),
            nn.Conv1d(ch, 128, 1),  # 通道升到 128
        )
        self.gru = nn.GRU(128, hidden, num_layers=1, batch_first=True)
        self.head_rhythm = nn.Linear(hidden, n_class)
        self.use_quality_head = use_quality_head
        if use_quality_head:
            self.head_quality = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def embed(self, x):  # x: (B, 1, L) → (B, hidden)
        h = self.msc(x)
        h = self.tcn(h)
        out, _ = self.gru(h.transpose(1, 2))
        return out[:, -1]

    def forward(self, x):
        z = self.embed(x)
        logits = self.head_rhythm(z)
        # 质量头输出 logit（配合 BCEWithLogits，AMP 安全）；推理时用 sigmoid
        q = self.head_quality(z).squeeze(-1) if self.use_quality_head else None
        return logits, q


class ClassBalancedFocalLoss(nn.Module):
    """Effective number of samples 加权的 focal loss（chatGPT建议3 第11节）。"""

    def __init__(self, counts: torch.Tensor, beta=0.999, gamma=2.0):
        super().__init__()
        counts = counts.float().clamp(min=1)
        effective = (1 - beta) / (1 - beta ** counts)
        w = effective / effective.sum() * len(counts)
        self.register_buffer("w", w)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        w_t = self.w.to(target.device)[target]
        loss = -w_t * (1 - p_t).clamp(max=1.0).pow(self.gamma) * logp_t
        return loss.mean()
