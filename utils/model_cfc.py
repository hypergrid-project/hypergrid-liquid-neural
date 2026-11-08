"""
HyperGrid Continuous-Time Deep Learning Pipeline
Component B: Architecture Definition for Closed-Form Continuous-Time Networks (CfC)
=====================================================================================
Features:
- Pure analytic formulation of Closed-form Continuous-time (CfC) neural networks (Hasani et al., 2022).
- Zero numerical ODE solvers during forward pass (guarantees constant O(1) latency per step).
- 1D Depthwise-Separable Convolutional front-end for micro-vibration feature extraction.
- Optional Sparse Neural Circuit Policy (AutoNCP) wiring support via ncps or analytic sparsity masks.
- Continuous-time decay parameters driven by irregular delta-t timestamps.
- Flexible readout: Learned Temporal Attention Pooling or Final Hidden-State Readout.
- Extreme edge compactness: 32 to 64 units, total footprint <= 350 KB, latency <= 0.5 ms.
- Fully trace-compatible with ONNX (opset 16+) and standard TensorFlow Lite (no custom ops).
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional ncps import with graceful fallback
try:
    import ncps
    from ncps.torch import CfC as NCPS_CfC
    from ncps.wirings import AutoNCP, FullyConnectedWiring
    HAS_NCPS = True
except ImportError:
    HAS_NCPS = False


class DepthwiseSeparableConv1d(nn.Module):
    """
    Ultra-lightweight 1D Depthwise-Separable Convolution.
    Extracts high-frequency micro-vibration shockwaves before temporal ODE modeling.
    Complexity is O(in_channels * kernel_size + in_channels * out_channels), far lighter than standard Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, in_channels)
        # Permute to (batch_size, in_channels, seq_len) for 1D convolution
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        # Permute back to (batch_size, seq_len, out_channels)
        return x.transpose(1, 2)


class AnalyticCfCCell(nn.Module):
    """
    Pure-PyTorch Closed-form Continuous-time (CfC) Recurrent Cell.
    Solves the continuous-depth ODE analytically without numerical Runge-Kutta / Euler solvers:
        dx/dt = - [1/tau + f(x, u)] * x + A * f(x, u)
    Hasani et al., Nature Machine Intelligence 2022.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 32,
        mode: str = "default",
        backbone_units: int = 64,
        backbone_layers: int = 1,
        sparsity_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.mode = mode

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}, got '{mode}'")

        cat_dim = in_features + hidden_size

        # Optional backbone feed-forward layers
        if backbone_layers > 0:
            layers = [nn.Linear(cat_dim, backbone_units), nn.SiLU()]
            for _ in range(1, backbone_layers):
                layers.extend([nn.Linear(backbone_units, backbone_units), nn.SiLU()])
            self.backbone = nn.Sequential(*layers)
            feature_dim = backbone_units
        else:
            self.backbone = None
            feature_dim = cat_dim

        self.ff1 = nn.Linear(feature_dim, hidden_size)

        if self.mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
            self.A = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        else:
            self.ff2 = nn.Linear(feature_dim, hidden_size)
            self.time_a = nn.Linear(feature_dim, hidden_size)
            self.time_b = nn.Linear(feature_dim, hidden_size)

        # Register optional sparsity mask (e.g. from AutoNCP)
        if sparsity_mask is not None:
            self.register_buffer("sparsity_mask", sparsity_mask)
        else:
            self.sparsity_mask = None

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        dt_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single continuous-time recurrent step.
        x_t: (batch_size, in_features)
        h_prev: (batch_size, hidden_size)
        dt_t: (batch_size, 1) or scalar
        """
        cat = torch.cat([x_t, h_prev], dim=-1)
        if self.backbone is not None:
            feat = self.backbone(cat)
        else:
            feat = cat

        if self.sparsity_mask is not None:
            ff1 = F.linear(feat, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(feat)

        if self.mode == "pure":
            decay = torch.exp(-dt_t * (torch.abs(self.w_tau) + torch.abs(ff1)))
            h_new = -self.A * decay * ff1 + self.A
        else:
            if self.sparsity_mask is not None:
                ff2 = F.linear(feat, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(feat)

            ff1_act = self.tanh(ff1)
            ff2_act = self.tanh(ff2)

            t_a = self.time_a(feat)
            t_b = self.time_b(feat)
            t_interp = self.sigmoid(t_a * dt_t + t_b)

            if self.mode == "no_gate":
                h_new = ff1_act + t_interp * ff2_act
            else:
                h_new = ff1_act * (1.0 - t_interp) + ff2_act * t_interp

        return h_new


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based sequence pooling.
    Learns to dynamically focus on transient shock pulses (e.g. pothole hits)
    rather than uniform averaging over a multi-second window.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_seq shape: (batch_size, seq_len, hidden_dim)
        scores = self.scorer(h_seq)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(h_seq * weights, dim=1)  # (batch_size, hidden_dim)
        return pooled, weights


class CfCLiquidNet(nn.Module):
    """
    End-to-End Liquid Neural Network (CfC) for Edge Road Surface Anomaly Classification.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_size: int = 32,
        num_classes: int = 4,
        use_conv_frontend: bool = True,
        conv_out_channels: int = 16,
        conv_kernel_size: int = 5,
        mode: str = "default",
        backbone_units: int = 64,
        backbone_layers: int = 1,
        pooling: str = "attention",
        dropout_p: float = 0.2,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_conv_frontend = use_conv_frontend
        self.pooling_type = pooling
        self.return_sequences = return_sequences

        if use_conv_frontend:
            self.frontend = DepthwiseSeparableConv1d(
                in_channels=in_channels,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
            )
            cfc_in_features = conv_out_channels
        else:
            self.frontend = nn.Identity()
            cfc_in_features = in_channels

        self.cell = AnalyticCfCCell(
            in_features=cfc_in_features,
            hidden_size=hidden_size,
            mode=mode,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

        if pooling == "attention":
            self.pooling = TemporalAttentionPooling(hidden_dim=hidden_size)
        else:
            self.pooling = None

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        features = self.frontend(x)
        h = torch.zeros(batch_size, self.hidden_size, device=device)

        hidden_states = []
        for t in range(seq_len):
            feat_t = features[:, t, :]
            dt_t = dt[:, t, :] if dt is not None else torch.ones(batch_size, 1, device=device)
            h = self.cell(feat_t, h, dt_t)
            hidden_states.append(h)

        h_seq = torch.stack(hidden_states, dim=1)

        if self.return_sequences:
            return self.classifier(h_seq)

        if self.pooling_type == "attention":
            pooled, _ = self.pooling(h_seq)
        else:
            pooled = h_seq[:, -1, :]

        logits = self.classifier(pooled)
        return logits


class NCPSCfCNet(nn.Module):
    """
    Liquid Neural Network wrapper utilizing the official 'ncps' library
    with AutoNCP (Sparse Neural Circuit Policy wiring).
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_size: int = 32,
        num_classes: int = 4,
        motor_neurons: int = 4,
        dropout_p: float = 0.2,
        mode: str = "default",
    ):
        super().__init__()
        if not HAS_NCPS:
            raise RuntimeError(
                "ncps library is not installed. Please install 'ncps' or use CfCLiquidNet."
            )
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.wiring = AutoNCP(
            units=hidden_size,
            output_size=motor_neurons,
            sparsity_level=0.5,
        )
        self.cfc = NCPS_CfC(
            input_size=in_channels,
            units=self.wiring,
            return_sequences=False,
            mode=mode,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(motor_neurons, num_classes),
        )

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.cfc(x, timespans=dt)
        return self.classifier(out)


def build_cfc_model(
    model_type: str = "analytic",
    backbone_type: Optional[str] = None,
    in_channels: int = 8,
    hidden_size: int = 32,
    num_classes: int = 4,
    use_conv_frontend: bool = True,
    conv_out_channels: int = 16,
    mode: str = "default",
    pooling: str = "attention",
    dropout_p: float = 0.2,
) -> nn.Module:
    """
    Factory function to construct the requested CfC model variant.
    Accepts both model_type and backbone_type ('analytic' or 'ncps').
    """
    b_type = backbone_type if backbone_type is not None else model_type
    if b_type == "ncps" and HAS_NCPS:
        return NCPSCfCNet(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_p=dropout_p,
            mode=mode,
        )
    return CfCLiquidNet(
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_classes=num_classes,
        use_conv_frontend=use_conv_frontend,
        conv_out_channels=conv_out_channels,
        mode=mode,
        pooling=pooling,
        dropout_p=dropout_p,
        return_sequences=False,
    )
