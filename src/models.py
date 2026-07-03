"""Forecasting models: LSTM, Transformer, and proposed Decomp-MTCFormer.

Only the third model (MTCFormer) is changed compared with the original project.
The public class name and build_model interface are kept unchanged, so existing
train_mtcformer.py can be used without modification.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled)


class MovingAverage1D(nn.Module):
    """Moving average along the temporal dimension for target decomposition.

    Input/Output shape: [B, T]. Edge values are repeated before pooling so that
    the output length is the same as the input length.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        if self.kernel_size <= 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        x_pad = torch.cat([front, x, end], dim=1).unsqueeze(1)  # [B, 1, T + 2*pad]
        return self.avg(x_pad).squeeze(1)


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal convolution block with adaptive branch gating."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv3 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2)
        self.dilated = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=2, dilation=2)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        xt = x.transpose(1, 2)
        branches = [
            F.gelu(self.conv3(xt)).transpose(1, 2),
            F.gelu(self.conv5(xt)).transpose(1, 2),
            F.gelu(self.dilated(xt)).transpose(1, 2),
        ]
        pooled = torch.cat([b.mean(dim=1) for b in branches], dim=-1)
        weights = self.gate(pooled)  # [B, 3]
        z = sum(weights[:, i].view(-1, 1, 1) * branches[i] for i in range(3))
        return self.norm(z)


class MTCFormer(nn.Module):
    """Decomp-MTCFormer: trend decomposition enhanced MTCFormer.

    Compared with the original MTCFormer, this version explicitly decomposes the
    historical target curve into a smooth trend and a residual component.

    - The target residual is written back to the first input channel and modeled
      by the multi-scale convolution + Transformer branch.
    - The smooth target trend is mapped to the future horizon by a trend branch.
    - The final prediction is the residual/global prediction plus a learnable
      weighted trend prediction.

    The external features, such as voltage, current, sub-metering and weather
    variables, are kept unchanged in the residual branch so that the model can
    still use all auxiliary variables.
    """

    def __init__(
        self,
        input_dim: int,
        input_len: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        decomp_kernel: int = 25,
    ):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.target_ma = MovingAverage1D(kernel_size=decomp_kernel)

        self.temporal = MultiScaleTemporalBlock(input_dim, d_model, dropout=dropout)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Residual/global branch: predicts future residual and nonlinear changes.
        self.global_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

        # Trend branch: maps the smoothed historical target trend to the future.
        self.trend_head = nn.Sequential(
            nn.LayerNorm(input_len),
            nn.Linear(input_len, horizon),
        )

        # Learnable fusion strength for the trend branch.
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]. The first channel is the scaled target global_active_power.
        target_hist = x[:, :, 0]                       # [B, T]
        trend_hist = self.target_ma(target_hist)       # [B, T]
        residual_hist = target_hist - trend_hist       # [B, T]

        # Keep exogenous variables unchanged, only replace target channel by residual.
        x_res = x.clone()
        x_res[:, :, 0] = residual_hist

        z = self.temporal(x_res)
        z = self.encoder(self.pos(z))
        residual_pred = self.global_head(z.mean(dim=1))
        trend_pred = self.trend_head(trend_hist)

        return residual_pred + torch.sigmoid(self.alpha) * trend_pred


def build_model(model_name: str, input_dim: int, input_len: int, horizon: int) -> nn.Module:
    name = model_name.lower()
    if name == "lstm":
        return LSTMForecast(input_dim=input_dim, horizon=horizon)
    if name == "transformer":
        return TransformerForecast(input_dim=input_dim, horizon=horizon)
    if name in {"mtcformer", "improved", "proposed"}:
        return MTCFormer(input_dim=input_dim, input_len=input_len, horizon=horizon)
    raise ValueError(f"Unknown model name: {model_name}")
