"""
Spectrapur backend API.

Endpoints
---------
GET  /api/health            -> {"status": "ok", "device": "..."}
POST /api/denoise           -> run the ResUNet1D denoiser on a spectrum.
                                Accepts either:
                                  - multipart/form-data with a "file" field
                                    (.csv / .txt / .tsv, 1 or 2 columns), or
                                  - application/json body: {"x": [...], "y": [...]}
                                    ("x" is optional; "y" alone is accepted)
POST /api/denoise/sample    -> run the demo on a built-in synthetic sample
                                spectrum (used by the "Try a sample" button).

Run locally:
    pip install -r requirements.txt
    python app.py

Deploy on Render:
    Start command:  gunicorn app:app --timeout 120
"""

from __future__ import annotations

import csv
import io
import os
import re

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from inference.denoiser import denoise_spectrum, get_model

app = Flask(__name__)

# Allow the deployed frontend (Vercel) plus local dev to call this API.
# Set ALLOWED_ORIGINS as a comma-separated env var in production, e.g.
#   ALLOWED_ORIGINS=https://spectrapur.vercel.app,http://localhost:5173
_allowed = os.getenv("ALLOWED_ORIGINS", "*")
_origins = "*" if _allowed.strip() == "*" else [o.strip() for o in _allowed.split(",")]
CORS(app, resources={r"/api/*": {"origins": _origins}})

MAX_POINTS = 20000
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_number(token: str) -> bool:
    return bool(_NUM_RE.match(token.strip()))


def parse_spectrum_text(raw_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse CSV/TSV/whitespace-delimited text into (x, y) arrays.

    Supports:
      - two columns: x,y per line
      - one column: y only (x becomes the point index)
      - an optional header row (skipped automatically if non-numeric)
      - comma, tab, or whitespace delimited
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("Uploaded file is empty.")

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    sniffed_delim = ","
    try:
        dialect = csv.Sniffer().sniff(lines[0], delimiters=",\t; ")
        sniffed_delim = dialect.delimiter
    except Exception:
        if "\t" in lines[0]:
            sniffed_delim = "\t"
        elif "," in lines[0]:
            sniffed_delim = ","
        else:
            sniffed_delim = None  # whitespace split

    rows = []
    for ln in lines:
        if sniffed_delim:
            parts = [p.strip() for p in ln.split(sniffed_delim) if p.strip() != ""]
        else:
            parts = ln.split()
        if not parts:
            continue
        rows.append(parts)

    # Drop a header row if its first cell isn't numeric
    if rows and not _is_number(rows[0][0]):
        rows = rows[1:]

    if not rows:
        raise ValueError("No numeric data rows found in file.")

    ncols = len(rows[0])
    xs, ys = [], []

    if ncols >= 2:
        for r in rows:
            if len(r) < 2 or not _is_number(r[0]) or not _is_number(r[1]):
                continue
            xs.append(float(r[0]))
            ys.append(float(r[1]))
    else:
        for r in rows:
            if not _is_number(r[0]):
                continue
            ys.append(float(r[0]))
        xs = list(range(len(ys)))

    if len(ys) < 8:
        raise ValueError("Need at least 8 valid numeric data points.")

    if len(ys) > MAX_POINTS:
        raise ValueError(f"Spectrum has {len(ys)} points; the limit is {MAX_POINTS}.")

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _make_sample_spectrum(n: int = 1200, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 4000, n)  # wavenumber-like axis (cm^-1)

    def gaussian(center, height, width):
        return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    clean = (
        gaussian(1050, 0.65, 25)
        + gaussian(1450, 0.45, 18)
        + gaussian(1650, 0.85, 22)
        + gaussian(2350, 0.30, 15)
        + gaussian(2920, 0.95, 30)
        + gaussian(3350, 0.55, 60)
    )
    baseline = 0.05 + 0.00004 * (x - 400) + 0.02 * np.sin(x / 900.0)
    noise = rng.normal(0, 0.02, size=n)
    spikes_idx = rng.choice(n, size=6, replace=False)
    spikes = np.zeros(n)
    spikes[spikes_idx] = rng.uniform(0.1, 0.25, size=6) * rng.choice([-1, 1], size=6)

    noisy = clean + baseline + noise + spikes
    return x, noisy


@app.get("/api/health")
def health():
    try:
        _, device = get_model()
        return jsonify({"status": "ok", "device": str(device)})
    except Exception as exc:  # pragma: no cover
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/api/denoise")
def denoise():
    try:
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            raw_text = f.read().decode("utf-8", errors="ignore")
            x, y = parse_spectrum_text(raw_text)
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            y = body.get("y")
            if y is None:
                return jsonify({"error": "JSON body must include a 'y' array."}), 400
            y = np.array(y, dtype=np.float64)
            x = body.get("x")
            x = np.array(x, dtype=np.float64) if x is not None else np.arange(len(y), dtype=np.float64)
            if len(y) < 8:
                return jsonify({"error": "Need at least 8 valid numeric data points."}), 400
            if len(y) > MAX_POINTS:
                return jsonify({"error": f"Spectrum has {len(y)} points; the limit is {MAX_POINTS}."}), 400
        else:
            return jsonify({"error": "Send a file upload ('file' field) or a JSON body with 'y'."}), 400

        result = denoise_spectrum(x, y)
        return jsonify({
            "x": result.x,
            "y_noisy": result.y_noisy,
            "y_denoised": result.y_denoised,
            "residual": result.residual,
            "stats": result.stats,
            "peaks_noisy": result.peaks_noisy,
            "peaks_denoised": result.peaks_denoised,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:  # pragma: no cover
        app.logger.exception("Denoise failed")
        return jsonify({"error": f"Internal error while denoising: {exc}"}), 500


@app.post("/api/denoise/sample")
def denoise_sample():
    try:
        x, y = _make_sample_spectrum()
        result = denoise_spectrum(x, y)
        return jsonify({
            "x": result.x,
            "y_noisy": result.y_noisy,
            "y_denoised": result.y_denoised,
            "residual": result.residual,
            "stats": result.stats,
            "peaks_noisy": result.peaks_noisy,
            "peaks_denoised": result.peaks_denoised,
        })
    except Exception as exc:  # pragma: no cover
        app.logger.exception("Sample denoise failed")
        return jsonify({"error": f"Internal error while denoising sample: {exc}"}), 500


if __name__ == "__main__":
    # Warm the model up on boot so the first request isn't slow.
    get_model()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
