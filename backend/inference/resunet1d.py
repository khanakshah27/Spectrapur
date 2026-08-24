import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.size(-1) == ref.size(-1):
        return x
    return F.interpolate(x, size=ref.size(-1), mode="linear", align_corners=False)


def _make_norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    if norm_type == "group":
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return nn.GroupNorm(groups, channels)
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm type: {norm_type}")


class ReflectionConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.pad = nn.ReflectionPad1d(pad)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class MedianFilter1D(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("MedianFilter1D kernel_size must be odd")
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        x_pad = F.pad(x, (self.pad, self.pad), mode="reflect")
        windows = x_pad.unfold(dimension=-1, size=self.kernel_size, step=1)
        return windows.median(dim=-1).values


class AdaptiveSpikeSuppressor1D(nn.Module):
    def __init__(
        self,
        median_kernel: int = 5,
        scale_kernel: int = 9,
        threshold: float = 2.5,
        blend: float = 0.85,
        edge_points: int = 96,
        edge_gain: float = 1.8,
    ):
        super().__init__()
        if scale_kernel % 2 == 0:
            raise ValueError("AdaptiveSpikeSuppressor1D scale_kernel must be odd")
        self.median = MedianFilter1D(kernel_size=median_kernel)
        self.scale_kernel = scale_kernel
        self.threshold = threshold
        self.blend = blend
        self.edge_points = edge_points
        self.edge_gain = edge_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        median_x = self.median(x)
        residual = x - median_x
        pad = self.scale_kernel // 2
        local_scale = F.avg_pool1d(
            F.pad(torch.abs(residual), (pad, pad), mode="reflect"),
            kernel_size=self.scale_kernel,
            stride=1,
        )
        local_scale = local_scale.clamp_min(1e-4)

        effective_threshold = torch.full_like(local_scale, self.threshold)
        edge_points = min(self.edge_points, x.size(-1))
        if edge_points > 0:
            effective_threshold[..., :edge_points] = effective_threshold[..., :edge_points] / self.edge_gain
            effective_threshold[..., -edge_points:] = effective_threshold[..., -edge_points:] / max(
                1.0, self.edge_gain * 0.75
            )

        clipped_residual = torch.clamp(
            residual,
            min=-effective_threshold * local_scale,
            max=effective_threshold * local_scale,
        )
        corrected = median_x + clipped_residual
        return (1.0 - self.blend) * x + self.blend * corrected


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(self.pool(x))
        return x * scale


class SpectralAttention1D(nn.Module):
    """Lightweight position-aware attention over the spectral axis."""

    def __init__(
        self,
        channels: int,
        target_len: int = 1868,
        focus_start: int = 104,
        focus_end: int = 726,
        hidden_channels: int | None = None,
        prior_strength: float = 0.0,
        max_gain: float = 0.45,
        norm_type: str = "group",
    ):
        super().__init__()
        hidden = hidden_channels or max(channels // 8, 16)
        self.target_len = max(1, int(target_len))
        self.focus_start = int(focus_start)
        self.focus_end = int(focus_end)
        self.prior_strength = float(prior_strength)
        self.max_gain = float(max_gain)

        self.feature_gate = nn.Sequential(
            ReflectionConv1d(channels, hidden, kernel_size=7),
            _make_norm(hidden, norm_type),
            nn.GELU(),
            ReflectionConv1d(hidden, 1, kernel_size=7),
        )
        self.coord_gate = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.feature_gate[-1].conv.weight)
        nn.init.zeros_(self.coord_gate[-1].weight)
        if self.coord_gate[-1].bias is not None:
            nn.init.zeros_(self.coord_gate[-1].bias)

    def _coordinate_features(self, x: torch.Tensor) -> torch.Tensor:
        b, _, length = x.shape
        coord = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype).view(1, 1, length)
        focus = torch.zeros_like(coord)
        start = int(round(max(0, self.focus_start) / self.target_len * length))
        end = int(round(max(self.focus_start + 1, self.focus_end) / self.target_len * length))
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        focus[..., start:end] = 1.0
        edge_distance = 1.0 - torch.abs(coord)
        return torch.cat([coord, focus, edge_distance], dim=1).expand(b, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self._coordinate_features(x)
        focus_prior = coords[:, 1:2]
        logits = self.feature_gate(x) + self.coord_gate(coords) + self.prior_strength * focus_prior
        gain = 1.0 + self.max_gain * torch.tanh(logits)
        return x * gain


class MultiScaleContext1D(nn.Module):
    def __init__(self, channels: int, norm_type: str = "group", use_se: bool = True):
        super().__init__()
        branch_channels = max(channels // 4, 16)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ReflectionConv1d(channels, branch_channels, kernel_size=3, dilation=1),
                    _make_norm(branch_channels, norm_type),
                    nn.GELU(),
                ),
                nn.Sequential(
                    ReflectionConv1d(channels, branch_channels, kernel_size=5, dilation=1),
                    _make_norm(branch_channels, norm_type),
                    nn.GELU(),
                ),
                nn.Sequential(
                    ReflectionConv1d(channels, branch_channels, kernel_size=3, dilation=2),
                    _make_norm(branch_channels, norm_type),
                    nn.GELU(),
                ),
                nn.Sequential(
                    ReflectionConv1d(channels, branch_channels, kernel_size=3, dilation=4),
                    _make_norm(branch_channels, norm_type),
                    nn.GELU(),
                ),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * len(self.branches), channels, kernel_size=1, bias=False),
            _make_norm(channels, norm_type),
            nn.GELU(),
        )
        self.se = SqueezeExcite1D(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = torch.cat([branch(x) for branch in self.branches], dim=1)
        fused = self.se(self.fuse(multi))
        return x + fused


class ResidualConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        dilation: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
        self.conv1 = ReflectionConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.norm1 = _make_norm(out_channels, norm_type)
        self.conv2 = ReflectionConv1d(out_channels, out_channels, kernel_size=3, dilation=1)
        self.norm2 = _make_norm(out_channels, norm_type)
        self.act = nn.GELU()
        self.se = SqueezeExcite1D(out_channels) if use_se else nn.Identity()
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


class AttentionGate1D(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv1d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = _align_to(gate, skip)
        attn = self.act(self.theta(skip) + self.phi(gate))
        attn = torch.sigmoid(self.psi(attn))
        return skip * attn


class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "group",
        use_skip_gate: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.proj = nn.Sequential(
            ReflectionConv1d(in_channels, out_channels, kernel_size=3),
            _make_norm(out_channels, norm_type),
            nn.GELU(),
        )
        self.skip_gate = (
            AttentionGate1D(skip_channels, out_channels, max(out_channels // 2, 8))
            if use_skip_gate
            else nn.Identity()
        )
        self.fuse = ResidualConvBlock1D(
            out_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            use_se=use_se,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.upsample(x))
        x = _align_to(x, skip)
        gated_skip = self.skip_gate(skip, x) if isinstance(self.skip_gate, AttentionGate1D) else skip
        return self.fuse(torch.cat([x, gated_skip], dim=1))


class DetailRefiner1D(nn.Module):
    def __init__(self, channels: int, norm_type: str = "group", use_se: bool = True):
        super().__init__()
        self.pre = nn.Sequential(
            ReflectionConv1d(channels + 2, channels, kernel_size=5),
            _make_norm(channels, norm_type),
            nn.GELU(),
        )
        self.block1 = ResidualConvBlock1D(channels, channels, norm_type=norm_type, dilation=1, use_se=use_se)
        self.block2 = ResidualConvBlock1D(channels, channels, norm_type=norm_type, dilation=2, use_se=use_se)
        self.block3 = ResidualConvBlock1D(channels, channels, norm_type=norm_type, dilation=3, use_se=use_se)
        self.out = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool1d(F.pad(raw_input, (2, 2), mode="reflect"), kernel_size=5, stride=1)
        highpass = raw_input - smooth
        x = torch.cat([features, raw_input, highpass], dim=1)
        x = self.pre(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out(x)


class PositionalEncoder1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[2].weight)
        if self.net[2].bias is not None:
            nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, l = x.shape
        coord = torch.linspace(-1.0, 1.0, l, device=x.device, dtype=x.dtype).view(1, 1, l).expand(b, -1, -1)
        return self.net(coord)


class DerivativeEncoder1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.pre = nn.Sequential(
            ReflectionConv1d(2, hidden, kernel_size=3),
            nn.GELU(),
        )
        self.out = ReflectionConv1d(hidden, channels, kernel_size=3)
        nn.init.zeros_(self.out.conv.weight)
        if self.out.conv.bias is not None:
            nn.init.zeros_(self.out.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = F.pad(x[..., 1:] - x[..., :-1], (1, 0), mode="replicate")
        d2 = F.pad(d1[..., 1:] - d1[..., :-1], (1, 0), mode="replicate")
        feats = torch.cat([d1, d2], dim=1)
        return self.out(self.pre(feats))


class LocalRefiner1D(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv1d(3, hidden_channels, kernel_size=5),
            nn.GELU(),
            ReflectionConv1d(hidden_channels, hidden_channels, kernel_size=3),
            nn.GELU(),
            ReflectionConv1d(hidden_channels, hidden_channels, kernel_size=3),
            nn.GELU(),
        )
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = F.pad(x[..., 1:] - x[..., :-1], (1, 0), mode="replicate")
        d2 = F.pad(d1[..., 1:] - d1[..., :-1], (1, 0), mode="replicate")
        feats = torch.cat([x, d1, d2], dim=1)
        return self.out(self.net(feats))


class StridedDownsample1D(nn.Module):
    """Trainable downsampling that preserves channel identity at initialization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.conv.weight, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        residual_learning: bool = True,
        norm_type: str = "group",
        use_skip_gates: bool = True,
        use_se: bool = True,
        use_input_median: bool = False,
        input_median_kernel: int = 5,
        input_median_blend: float = 0.25,
        use_spike_suppressor: bool = False,
        spike_suppressor_threshold: float = 2.5,
        spike_suppressor_blend: float = 0.85,
        spike_suppressor_edge_points: int = 96,
        spike_suppressor_edge_gain: float = 1.8,
        use_multiscale_context: bool = False,
        use_detail_head: bool = False,
        use_positional_bias: bool = False,
        use_derivative_bias: bool = False,
        use_local_refiner: bool = False,
        use_spectral_attention: bool = False,
        target_len: int = 1868,
        attention_focus_start: int = 104,
        attention_focus_end: int = 726,
        spectral_attention_prior: float = 0.0,
        use_strided_downsample: bool = False,
    ):
        super().__init__()
        self.residual_learning = residual_learning
        self.norm_type = norm_type
        self.use_skip_gates = use_skip_gates
        self.use_se = use_se
        self.use_input_median = use_input_median
        self.input_median_blend = input_median_blend
        self.use_spike_suppressor = use_spike_suppressor
        self.use_multiscale_context = use_multiscale_context
        self.use_detail_head = use_detail_head
        self.use_positional_bias = use_positional_bias
        self.use_derivative_bias = use_derivative_bias
        self.use_local_refiner = use_local_refiner
        self.use_spectral_attention = use_spectral_attention
        self.use_strided_downsample = use_strided_downsample
        self.input_median = MedianFilter1D(kernel_size=input_median_kernel) if use_input_median else nn.Identity()
        self.spike_suppressor = (
            AdaptiveSpikeSuppressor1D(
                median_kernel=input_median_kernel,
                scale_kernel=max(input_median_kernel + 4, 7) | 1,
                threshold=spike_suppressor_threshold,
                blend=spike_suppressor_blend,
                edge_points=spike_suppressor_edge_points,
                edge_gain=spike_suppressor_edge_gain,
            )
            if use_spike_suppressor
            else nn.Identity()
        )

        self.enc1 = ResidualConvBlock1D(in_channels, base_channels, norm_type=norm_type, use_se=use_se)
        self.spectral_attn1 = (
            SpectralAttention1D(
                base_channels,
                target_len=target_len,
                focus_start=attention_focus_start,
                focus_end=attention_focus_end,
                prior_strength=spectral_attention_prior,
                norm_type=norm_type,
            )
            if use_spectral_attention
            else nn.Identity()
        )
        self.positional_encoder = PositionalEncoder1D(base_channels) if use_positional_bias else nn.Identity()
        self.derivative_encoder = DerivativeEncoder1D(base_channels) if use_derivative_bias else nn.Identity()
        self.enc2 = ResidualConvBlock1D(base_channels, base_channels * 2, norm_type=norm_type, use_se=use_se)
        self.spectral_attn2 = (
            SpectralAttention1D(
                base_channels * 2,
                target_len=target_len,
                focus_start=attention_focus_start,
                focus_end=attention_focus_end,
                prior_strength=spectral_attention_prior,
                norm_type=norm_type,
            )
            if use_spectral_attention
            else nn.Identity()
        )
        self.enc3 = ResidualConvBlock1D(base_channels * 2, base_channels * 4, norm_type=norm_type, use_se=use_se)
        self.spectral_attn3 = (
            SpectralAttention1D(
                base_channels * 4,
                target_len=target_len,
                focus_start=attention_focus_start,
                focus_end=attention_focus_end,
                prior_strength=spectral_attention_prior,
                norm_type=norm_type,
            )
            if use_spectral_attention
            else nn.Identity()
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down1 = StridedDownsample1D(base_channels) if use_strided_downsample else self.pool
        self.down2 = StridedDownsample1D(base_channels * 2) if use_strided_downsample else self.pool
        self.down3 = StridedDownsample1D(base_channels * 4) if use_strided_downsample else self.pool

        self.bottleneck1 = ResidualConvBlock1D(
            base_channels * 4,
            base_channels * 8,
            norm_type=norm_type,
            dilation=2,
            use_se=use_se,
        )
        self.bottleneck2 = ResidualConvBlock1D(
            base_channels * 8,
            base_channels * 8,
            norm_type=norm_type,
            dilation=4,
            use_se=use_se,
        )
        self.bottleneck3 = ResidualConvBlock1D(
            base_channels * 8,
            base_channels * 8,
            norm_type=norm_type,
            dilation=8,
            use_se=use_se,
        )
        self.multiscale_context = (
            MultiScaleContext1D(base_channels * 8, norm_type=norm_type, use_se=use_se)
            if use_multiscale_context
            else nn.Identity()
        )

        self.dec2 = UpBlock1D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            norm_type=norm_type,
            use_skip_gate=use_skip_gates,
            use_se=use_se,
        )
        self.dec1 = UpBlock1D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            norm_type=norm_type,
            use_skip_gate=use_skip_gates,
            use_se=use_se,
        )
        self.dec0 = UpBlock1D(
            base_channels * 2,
            base_channels,
            base_channels,
            norm_type=norm_type,
            use_skip_gate=use_skip_gates,
            use_se=use_se,
        )

        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        self.detail_refiner = (
            DetailRefiner1D(base_channels, norm_type=norm_type, use_se=use_se)
            if use_detail_head
            else nn.Identity()
        )
        self.local_refiner = LocalRefiner1D(hidden_channels=max(base_channels // 2, 16)) if use_local_refiner else nn.Identity()
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        if self.use_spike_suppressor:
            x = self.spike_suppressor(x)
        if self.use_input_median:
            median_x = self.input_median(x)
            x = (1.0 - self.input_median_blend) * x + self.input_median_blend * median_x

        e1 = self.enc1(x)
        if self.use_positional_bias:
            e1 = e1 + self.positional_encoder(e1)
        if self.use_derivative_bias:
            e1 = e1 + self.derivative_encoder(residual_input)
        e1 = self.spectral_attn1(e1)
        e2 = self.enc2(self.down1(e1))
        e2 = self.spectral_attn2(e2)
        e3 = self.enc3(self.down2(e2))
        e3 = self.spectral_attn3(e3)

        b = self.bottleneck1(self.down3(e3))
        b = self.bottleneck2(b)
        b = self.bottleneck3(b)
        b = self.multiscale_context(b)

        d2 = self.dec2(b, e3)
        d1 = self.dec1(d2, e2)
        d0 = self.dec0(d1, e1)

        pred = self.out_conv(d0)
        if self.use_detail_head:
            pred = pred + self.detail_refiner(d0, residual_input)
        if self.residual_learning:
            output = residual_input - pred
        else:
            output = pred
        if self.use_local_refiner:
            output = output + self.local_refiner(residual_input)
        return output
