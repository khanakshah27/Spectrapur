# Spectrapur

A web app for denoising raw spectral data, built on the **ResUNet1D** model from
[`nabhya8013/spectral_denoise`](https://github.com/nabhya8013/spectral_denoise).

Upload a spectrum → get back a denoised, baseline-corrected version, an overlay
chart, a residual (noise-removed) chart, noise/SNR/baseline metrics, and a
detected-peaks table — all in the browser.

```
Spectrapur/
├── backend/                     Flask API + model (deploy to Render)
│   ├── app.py                   Routes: /api/health, /api/denoise, /api/denoise/sample
│   ├── inference/
│   │   ├── resunet1d.py         Model architecture (copied verbatim from the source repo)
│   │   ├── denoiser.py          Pre/post-processing, inference, analysis metrics
│   │   └── model/
│   │       └── resunet1d_single_stage_final.pth   Pretrained weights (~31 MB)
│   ├── requirements.txt
│   ├── Procfile
│   └── .gitignore
├── frontend/                    Static site (deploy to Vercel)
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── config.js            <- set your backend URL here
│       └── app.js
├── sample_data/
│   └── sample_spectrum.csv      A synthetic demo spectrum you can upload to test the app
└── render.yaml                  Render Blueprint (optional one-click deploy)
```

## How the model is used

The checkpoint was inspected directly to recover its exact configuration
(base_channels=64, with skip gates, squeeze-excite, multiscale context,
detail head, positional + derivative bias, local refiner, and spectral
attention all enabled — see `backend/inference/denoiser.py` for the full
config and reasoning). The state dict loads with `strict=True` and no
missing/unexpected keys.

Because the network expects a fixed-length input (1868 points, per the
original repo), the backend:

1. Resamples the incoming spectrum to 1868 points (FFT resampling).
2. Normalizes it using its own mean/std (a single-spectrum stand-in for the
   "joint z-score" scheme used at training time, since there's no clean
   reference spectrum available at inference time).
3. Runs the model.
4. De-normalizes and resamples the result back to your original number of
   points, so it lines up with your original x-axis.

**Note on domain fit:** the model was trained on FTIR-style spectra. Results
will be best on similar data (a fingerprint-region spectrum with a smooth
baseline and Gaussian/Lorentzian-like peaks). The built-in "sample spectrum"
is a synthetic stand-in for demoing the UI, not a guarantee of the model's
real-world performance on arbitrary data.

## Running locally

**Backend:**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → serving on http://localhost:5000
```

**Frontend:** just open `frontend/index.html` in a browser, or serve it:
```bash
cd frontend
python3 -m http.server 5173
# → http://localhost:5173
```
`frontend/js/config.js` already points to `http://localhost:5000` when the
page is loaded from `localhost`, so local dev works with no changes.

## Deploying

### Backend → Render

1. Push this repo to GitHub.
2. In Render: **New → Web Service**, connect the repo.
   - If using the included `render.yaml`, Render will pick up the config
     automatically (Blueprint deploy). Otherwise set manually:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --timeout 120 --workers 1 --threads 4 --bind 0.0.0.0:$PORT`
3. Add an environment variable `ALLOWED_ORIGINS` set to your Vercel URL once
   you have it (e.g. `https://spectrapur.vercel.app`), or leave as `*` while
   testing.
4. Deploy. Note the resulting URL, e.g. `https://spectrapur-api.onrender.com`.

> The free Render tier spins down when idle — the first request after a
> period of inactivity can take 30–60s while it wakes up and loads the model.
> This is normal; the "Denoising…" state on the button will just sit longer
> on that first call.

### Frontend → Vercel

1. Edit `frontend/js/config.js` and set `API_BASE_URL` to your deployed
   Render backend URL.
2. In Vercel: **New Project**, import the repo, set the **root directory**
   to `frontend`, framework preset "Other" (static site) — no build command
   needed.
3. Deploy. Then go back to Render's `ALLOWED_ORIGINS` env var and set it to
   your live Vercel URL for tighter CORS.

## API reference

`POST /api/denoise`
- `multipart/form-data` with a `file` field (`.csv`/`.txt`/`.tsv`), **or**
- `application/json`: `{"x": [...], "y": [...]}` (`x` optional)

Response:
```json
{
  "x": [...], "y_noisy": [...], "y_denoised": [...], "residual": [...],
  "stats": { "noise_reduction_pct": 0.0, "snr_db_before": 0.0, "...": "..." },
  "peaks_noisy": [{"x":0,"y":0,"index":0}],
  "peaks_denoised": [{"x":0,"y":0,"index":0}]
}
```

`POST /api/denoise/sample` — same response shape, using a built-in synthetic
demo spectrum (no body needed).

`GET /api/health` — `{"status": "ok", "device": "cpu"}`

## Credits

Model architecture and weights: [nabhya8013/spectral_denoise](https://github.com/nabhya8013/spectral_denoise)
(ResUNet1D, released under that repository's terms).
