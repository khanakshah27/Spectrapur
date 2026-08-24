"""
Spectrapur inference service.

Wraps the ResUNet1D spectral-denoising model (from nabhya8013/spectral_denoise)
with the pre/post-processing needed to run it on an arbitrary user-supplied
spectrum, plus a set of extra analysis metrics for the web app.

Model facts (recovered from the released checkpoint + repo config,
results/resunet_single_stage_final.json):
  - target_len          = 1868         (network is trained on this fixed length)
  - normalization        = "joint_zscore"
  - base_channels        = 64
  - residual_learning    = True   (network predicts a correction, output = input - pred)
  - all optional submodules (skip gates, SE, multiscale context, detail head,
    positional bias, derivative bias, local refiner, spectral attention) = ON
  - spike suppressor / input median = OFF

Because a real deployment will receive spectra of arbitrary length and on an
arbitrary intensity scale (not necessarily matching the training distribution),
we:
  1. Resample the incoming spectrum to the network's fixed length (1868) using
     FFT resampling (scipy.signal.resample), matching the approach shown in the
     model repo's own inference example.
  2. Normalize using the spectrum's own mean/std (a self "z-score"), since at
     inference time there is no paired clean spectrum to compute joint
     statistics from -- this is the practical single-spectrum analogue of the
     "joint_zscore" scheme used at training time.
  3. Run the model.
  4. De-normalize back with the same mean/std.
  5. Resample the result back to the caller's original number of points, so the
     denoised curve lines up with the original x-axis.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import resample, savgol_filter, find_peaks

from .resunet1d import ResUNet1D

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_MODEL_DIR, "model", "resunet1d_single_stage_final.pth")

TARGET_LEN = 1868
FOCUS_START = 104
FOCUS_END = 726

_lock = threading.Lock()
_model = None
_device = None


def _build_model() -> ResUNet1D:
    return ResUNet1D(
        base_channels=64,
        residual_learning=True,
        norm_type="group",
        use_skip_gates=True,
        use_se=True,
        use_input_median=False,
        use_spike_suppressor=False,
        use_multiscale_context=True,
        use_detail_head=True,
        use_positional_bias=True,
        use_derivative_bias=True,
        use_local_refiner=True,
        use_spectral_attention=True,
        target_len=TARGET_LEN,
        attention_focus_start=FOCUS_START,
        attention_focus_end=FOCUS_END,
    )


def get_model(weights_path: str | None = None) -> tuple[ResUNet1D, torch.device]:
    """Lazily load and cache the model (thread-safe, loaded once per process)."""
    global _model, _device
    with _lock:
        if _model is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _build_model()
            path = weights_path or _DEFAULT_WEIGHTS
            state_dict = torch.load(path, map_location=_device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(_device)
            _model = model
        return _model, _device


@dataclass
class DenoiseResult:
    x: list
    y_noisy: list
    y_denoised: list
    residual: list
    stats: dict
    peaks_noisy: list
    peaks_denoised: list


def _resample_1d(y: np.ndarray, new_len: int) -> np.ndarray:
    if len(y) == new_len:
        return y.astype(np.float32)
    return resample(y, new_len).astype(np.float32)


def _safe_savgol(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 7:
        return y.copy()
    window = min(31, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    poly = 3 if window > 3 else 2
    try:
        return savgol_filter(y, window_length=window, polyorder=poly)
    except Exception:
        return y.copy()


def _noise_std(y: np.ndarray) -> float:
    """Estimate a noise floor by looking at the residual against a smooth trend."""
    smooth = _safe_savgol(y)
    return float(np.std(y - smooth))


def _snr_db(y: np.ndarray) -> float:
    smooth = _safe_savgol(y)
    signal_power = float(np.var(smooth))
    noise_power = float(np.var(y - smooth)) + 1e-12
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def _baseline_drift(y: np.ndarray, edge_frac: float = 0.05) -> float:
    n = len(y)
    edge = max(3, int(n * edge_frac))
    start_level = float(np.mean(y[:edge]))
    end_level = float(np.mean(y[-edge:]))
    return float(end_level - start_level)


def _detect_peaks(x: np.ndarray, y: np.ndarray, max_peaks: int = 25) -> list:
    if len(y) < 5:
        return []
    span = float(np.ptp(y)) or 1.0
    peak_idx, props = find_peaks(y, prominence=span * 0.03, distance=max(2, len(y) // 200))
    if len(peak_idx) == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    peak_idx = peak_idx[order]
    sort_by_x = np.argsort(peak_idx)
    peak_idx = peak_idx[sort_by_x]
    return [
        {"x": float(x[i]), "y": float(y[i]), "index": int(i)}
        for i in peak_idx
    ]


def denoise_spectrum(x: np.ndarray, y: np.ndarray, weights_path: str | None = None) -> DenoiseResult:
    """Run the full pipeline on a single (x, y) spectrum of arbitrary length."""
    x = np.asarray(x, dtype=np.float64)
    y_raw = np.asarray(y, dtype=np.float64)
    n_original = len(y_raw)

    if n_original < 8:
        raise ValueError("Spectrum needs at least 8 data points.")

    model, device = get_model(weights_path)

    # 1. Resample to network's fixed length
    y_model_in = _resample_1d(y_raw, TARGET_LEN)

    # 2. Self z-score normalize
    mean = float(np.mean(y_model_in))
    std = float(np.std(y_model_in)) + 1e-8
    y_norm = (y_model_in - mean) / std

    # 3. Inference
    tensor_in = torch.from_numpy(y_norm.astype(np.float32)).view(1, 1, -1).to(device)
    with torch.no_grad():
        tensor_out = model(tensor_in)
    y_out_norm = tensor_out.squeeze().cpu().numpy()

    # 4. De-normalize
    y_out = y_out_norm * std + mean

    # 5. Resample back to the caller's original number of points
    y_denoised = _resample_1d(y_out, n_original)

    residual = (y_raw - y_denoised).tolist()

    noise_before = _noise_std(y_raw)
    noise_after = _noise_std(y_denoised)
    noise_reduction_pct = 0.0
    if noise_before > 1e-9:
        noise_reduction_pct = float(max(0.0, (1.0 - (noise_after / noise_before)) * 100.0))

    stats = {
        "n_points": n_original,
        "mean_before": float(np.mean(y_raw)),
        "mean_after": float(np.mean(y_denoised)),
        "std_before": float(np.std(y_raw)),
        "std_after": float(np.std(y_denoised)),
        "noise_std_before": noise_before,
        "noise_std_after": noise_after,
        "noise_reduction_pct": noise_reduction_pct,
        "snr_db_before": _snr_db(y_raw),
        "snr_db_after": _snr_db(y_denoised),
        "baseline_drift_before": _baseline_drift(y_raw),
        "baseline_drift_after": _baseline_drift(y_denoised),
        "max_abs_correction": float(np.max(np.abs(residual))) if residual else 0.0,
    }

    peaks_noisy = _detect_peaks(x, y_raw)
    peaks_denoised = _detect_peaks(x, y_denoised)

    return DenoiseResult(
        x=x.tolist(),
        y_noisy=y_raw.tolist(),
        y_denoised=y_denoised.tolist(),
        residual=residual,
        stats=stats,
        peaks_noisy=peaks_noisy,
        peaks_denoised=peaks_denoised,
    )
